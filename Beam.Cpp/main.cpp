#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <array>
#include "Global.h"
#include "Timer.h"
#include "Beam.h"

using namespace Beam;

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
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(current_lr));

    std::array<Losses, adam_epochs / adam_epochs_diff> all_losses;

    std::cout << "\n[Phase 1] Adam Training...\n";

    for (int epoch = 0; epoch < adam_epochs; ++epoch) {
        try {
            optimizer.zero_grad();

            auto losses = compute_losses(*model, physics_input);

            // Check for NaN or Inf values in loss
            if (torch::isnan(losses.total).any().item<bool>() || torch::isinf(losses.total).any().item<bool>()) {
                std::cerr << "Adam Warning: Loss is NaN or Inf in epoch " << epoch << std::endl;

                // Halve learning rate and update optimizer
                current_lr *= 0.5f;
                std::cerr << "Adjusting learning rate to " << current_lr << std::endl;
                for (auto& param_group : optimizer.param_groups()) {
                    auto& options = static_cast<torch::optim::AdamOptions&>(param_group.options());
                    options.lr(current_lr);
                }
                continue;
            }

            losses.total.backward({}, Global::keep_graph);
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);  // Gradient clipping
            optimizer.step();

            // Periodic logging
            if (epoch % 200 == 0) {
                all_losses[epoch / adam_epochs_diff] = losses;
                std::cout << "Epoch " << epoch << "/" << adam_epochs
                    << " - Total: " << losses.total.item<float>()
                    << " | Physics: " << losses.physics.item<float>()
                    << " | Boundary: " << losses.boundary.item<float>()
                    << " | L2 term: " << losses.l2_reg.item<float>()
                    << " - LR: " << current_lr << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cout << "Error during Adam epoch " << epoch << ": " << e.what() << std::endl;
        }
    }

    // === PHASE 2: LBFGS optimizer (fine-tuning) ===
    std::cout << "\n[Phase 2] LBFGS Finetuning...\n";

    torch::optim::LBFGS lbfgs(model->parameters(),
        torch::optim::LBFGSOptions(1.0)
        .max_iter(20)
        .tolerance_grad(1e-7)
        .tolerance_change(1e-9)
        .history_size(100));

    int lbfgs_epochs = 50;
    current_lr *= 0.5f;

    for (int epoch = 0; epoch < lbfgs_epochs; ++epoch) {
        try {
            auto closure = [&]() -> torch::Tensor {
                lbfgs.zero_grad();

                auto loss = compute_losses(*model, physics_input).total;

                // NaN/Inf check
                if (torch::isnan(loss).any().item<bool>() || torch::isinf(loss).any().item<bool>()) {
                    std::cerr << "LBFGS Warning: Loss is NaN or Inf in epoch " << epoch << std::endl;

                    current_lr *= 0.5f;
                    std::cerr << "Adjusting learning rate to " << current_lr << std::endl;
                    for (auto& param_group : lbfgs.param_groups()) {
                        auto& options = static_cast<torch::optim::LBFGSOptions&>(param_group.options());
                        options.lr(current_lr);
                    }
                    return torch::tensor(1.0f, torch::requires_grad(true));  // Dummy loss
                }

                loss.backward({}, Global::keep_graph);
                return loss;
                };

            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);  // Gradient clipping
            torch::Tensor loss = lbfgs.step(closure);

            if (epoch % 10 == 0) {
                std::cout << "LBFGS Epoch " << epoch << "/" << lbfgs_epochs
                    << " - Loss: " << loss.item<float>() << " - Learning Rate: " << current_lr << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cout << "Error during LBFGS epoch " << epoch << ": " << e.what() << std::endl;
        }
    }

    // === Final output ===
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\nTraining completed in " << duration.count() << " seconds\n";

    model->eval();  // Final evaluation mode
    visualize_solution(*model);

    return 0;
}
