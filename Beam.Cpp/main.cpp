#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <array>
#include <span>
#include "Global.h"
#include "Timer.h"
#include "Beam.h"

using namespace Beam;

struct OptimizerSupplements {
    uint16_t epochs{};
    uint16_t epochs_diff{};
    float learning_rate{};
    bool call_closure{ false };
};

/**
 * @brief Trains a Physics-Informed Neural Network (PINN) model using a given optimizer.
 *
 * This function supports both first-order optimizers (e.g., Adam) and second-order
 * optimizers (e.g., LBFGS) by passing a closure function that recomputes the forward
 * and backward passes on each optimizer step. It includes loss monitoring, gradient
 * clipping, learning rate adjustment on NaN/Inf detection, and periodic logging.
 *
 * @tparam OptimType The optimizer type (e.g., torch::optim::Adam, torch::optim::LBFGS).
 * @tparam OptimOptionType The optimizer's option type (e.g., torch::optim::AdamOptions).
 * @tparam N The number of stored loss snapshots (epochs / epochs_diff).
 *
 * @param model Reference to the PINN model to be trained.
 * @param physics_input The physics-based training data tensor.
 * @param optimizer Reference to the optimizer instance.
 * @param optSupplements Struct containing optimizer configuration details such as
 *        number of epochs, learning rate, and whether to call closure directly before step().
 *
 * @return std::array<Losses, N> Containing logged losses at each logging interval.
 */
template<typename OptimType, typename OptimOptionType, uint16_t N>
std::array<Losses, N> train(PINN& model, const torch::Tensor& physics_input, OptimType& optimizer, const OptimizerSupplements& optSupplements) {
    auto current_lr{ optSupplements.learning_rate };
    std::array<Losses, N> all_losses;
    for (int epoch = 0; epoch < optSupplements.epochs; ++epoch) {
        try {
            Beam::Losses losses{};
            auto closure = [&]() -> torch::Tensor {
                optimizer.zero_grad();

                losses = compute_losses(model, physics_input);
                auto loss = losses.total;

                // NaN/Inf check
                if (torch::isnan(loss).any().item<bool>() || torch::isinf(loss).any().item<bool>()) {
                    std::cerr << "Optimizer Warning: Loss is NaN or Inf in epoch " << epoch << std::endl;
                    current_lr *= 0.5f; // Halve learning rate and update optimizer
                    std::cerr << "Adjusting learning rate to " << current_lr << std::endl;
                    for (auto& param_group : optimizer.param_groups()) {
                        auto& options = static_cast<OptimOptionType&>(param_group.options());
                        options.lr(current_lr);
                    }
                    return torch::tensor(1.0f, torch::requires_grad(true));  // Dummy loss
                }

                loss.backward({}, Global::keep_graph);
                return loss;
            };

            if (optSupplements.call_closure) {
                auto loss = closure();
            }
            
            torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);
            optimizer.step(closure);

            // Periodic logging
            if (epoch % optSupplements.epochs_diff == 0) {
                all_losses[epoch / optSupplements.epochs_diff] = losses;
                std::cout << "Epoch " << epoch << "/" << optSupplements.epochs
                    << " - Total: " << losses.total.item<float>()
                    << " | Physics: " << losses.physics.item<float>()
                    << " | Boundary: " << losses.boundary.item<float>()
                    << " | L2 term: " << losses.l2_reg.item<float>()
                    << " - Learning rate: " << current_lr << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cout << "Error during Optimizer epoch " << epoch << ": " << e.what() << std::endl;
        }
    }
    return all_losses;
}

/**
 * @brief Main routine: trains a Physics-Informed Neural Network (PINN) for the Euler-Bernoulli beam equation.
 *
 * Training phases:
 *  - Phase 1: Adam optimizer for coarse fitting
 *  - Phase 2: LBFGS optimizer for fine-tuning
 *
 * Handles device selection, adaptive learning rate, gradient clipping, and logging.
 *
 * @return int Exit code (0 = success)
 */
int main() {
    Helper::Timer tim;
    std::cout << "Physics-Informed Neural Network for Beam Equation\n";
    std::cout << "====================================================\n\n";

    // --- Device selection (CPU vs. CUDA) ---
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA available - using GPU\n";
    }
    else {
        std::cout << "CUDA NOT available - using CPU\n";
    }

    // Create model instance and move to selected device
    auto model = std::make_unique<PINN>();
    model->to(device);

    // Generate physics training data and move to device
    auto physics_input = generate_training_data(100);
    physics_input = physics_input.to(device);

    // initial learning rate
    float current_lr = 0.1f;
    auto start_time = std::chrono::steady_clock::now();

    // === PHASE 1: Adam optimizer (coarse training) ===
    std::cout << "\n[Phase 1] Adam Training...\n";
    constexpr OptimizerSupplements adam_supplements{ .epochs = 500, .epochs_diff = 100, .learning_rate = 0.1f, .call_closure = true };
    constexpr uint16_t adam_losses_size{ adam_supplements.epochs / adam_supplements.epochs_diff };
    try
    {
        torch::optim::Adam adam_optimizer(model->parameters(), torch::optim::AdamOptions(current_lr));
        std::array<Losses, adam_losses_size> adam_losses = train<torch::optim::Adam, torch::optim::AdamOptions, adam_losses_size>(*model, physics_input, adam_optimizer, adam_supplements);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    
    // === PHASE 2: LBFGS optimizer (fine-tuning) ===
    std::cout << "\n[Phase 2] LBFGS Finetuning...\n";
    constexpr OptimizerSupplements lbfgs_supplements{ .epochs = 50, .epochs_diff = 10, .learning_rate = 0.05f, .call_closure = false };
    constexpr uint16_t lbfgs_losses_size{ lbfgs_supplements.epochs / lbfgs_supplements.epochs_diff };
    try
    {
        torch::optim::LBFGS lbfgs_optimizer(model->parameters(),
            torch::optim::LBFGSOptions(1.0)
            .max_iter(20)
            .tolerance_grad(1e-7)
            .tolerance_change(1e-9)
            .history_size(100));
        
        std::array<Losses, lbfgs_losses_size> lbfgs_losses = train<torch::optim::LBFGS, torch::optim::LBFGSOptions, lbfgs_losses_size>(*model, physics_input, lbfgs_optimizer, lbfgs_supplements);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    // === Final output ===
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\nTraining completed in " << duration.count() << " seconds\n";

    model->eval();  // Final evaluation mode
    visualize_solution(*model);

    return 0;
}
