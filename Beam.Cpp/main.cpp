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
    Helper::Timer tim("MAIN TIMER");
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
        Helper::Timer tim("ADAM TIMER");
        torch::optim::Adam adam_optimizer(model->parameters(), torch::optim::AdamOptions(current_lr));
        std::array<Losses, adam_losses_size> adam_losses = 
            train<torch::optim::Adam, torch::optim::AdamOptions, adam_losses_size>
            (*model, physics_input, adam_optimizer, adam_supplements, 15, 5e-6f);
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
        Helper::Timer tim("LBFGS TIMER");
        torch::optim::LBFGS lbfgs_optimizer(model->parameters(),
            torch::optim::LBFGSOptions(1.0)
            .max_iter(20)
            .tolerance_grad(1e-7)
            .tolerance_change(1e-9)
            .history_size(100));
        
        std::array<Losses, lbfgs_losses_size> lbfgs_losses = 
            train<torch::optim::LBFGS, torch::optim::LBFGSOptions, lbfgs_losses_size>
            (*model, physics_input, lbfgs_optimizer, lbfgs_supplements, 20, 1e-7f);
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
