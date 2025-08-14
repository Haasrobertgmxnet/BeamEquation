#pragma once

#include <torch/torch.h>
#include "Pinn.h"
#include "Global.h"
#include "Losses.h"

namespace Beam {

    /**
     * @brief Struct to hold different loss components.
     */
    //struct LossTerms {
    //    //PINN& model;
    //    torch::Tensor total;        // Sum of all losses
    //    torch::Tensor physics;      // Loss of deviations from the physical law F(u) = 0
    //    torch::Tensor boundary;     // Loss of deviations from the boundary conditions on u
    //    torch::Tensor l2_reg;       // Penalty term, if Tikhonov L2 regularisation is active
    //};

    /**
     * @brief Struct to hold different optimizer adjustments.
     */
    struct OptimizerSupplements {
        uint16_t epochs{};          // Number of epochs
        uint16_t epochs_diff{};     // Number of epochs for doing a logging action
        float learning_rate{};      // Initial learning rate
        bool call_closure{ false }; // Flag, if the closure lambda has to be called (ADAM) or not (LBFGS)
    };


    // Number of epochs for training with Adam optimizer
    constexpr auto adam_epochs = uint16_t{ 500 };
    constexpr auto adam_epochs_diff = uint16_t{ 100 };

    /**
     * @brief Generates random 1D training points in the interval [0, 1].
     * @param n_points Number of training samples
     * @return Tensor: input x in R^n
     */
    [[nodiscard]]
    torch::Tensor generate_training_data(const int n_points = 100) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
        auto x = torch::rand({ n_points, 1 }, options);  // Random points in [0, 1]
        return x;
    }

    /**
     * @brief Visualizes the solution u(x) in R of the trained PINN model on [0, 1].
     *
     * Calculates u(x) for the x values on a grid.
     *
     * @param model The trained PINN model (inference mode is set internally)
     * @param grid_size Number of equally spaced evaluation points in [0, 1]
     * @return torch::tensor containing u(x)
     */
    [[nodiscard]]
    torch::Tensor model_forward_wrapper(PINN& model, const int grid_size = 20) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor x_tensor_ = torch::linspace(0.0, 1.0, grid_size, options).clone();

        torch::Tensor u_pred;
        try {
            auto x_tensor = x_tensor_.unsqueeze(1);
            u_pred = model.forward({ x_tensor });
        }
        catch (const c10::Error& e) {
            std::cerr << "Error during forward: " << e.what() << "\n";
            return u_pred;
        }
        return u_pred;
    }

    /**
     * @brief Visualizes the solution u(x) in R of the trained PINN model on [0, 1].
     *
     * Prints x and u(x) values on a grid to the console in tabular form.
     *
     * @param model The trained PINN model (inference mode is set internally)
     * @param grid_size Number of equally spaced evaluation points in [0, 1]
     */
    void visualize_solution(PINN& model, const int grid_size = 20) {
        torch::Tensor u_pred = model_forward_wrapper(model, grid_size);
        std::cout << "u_pred:\n" << u_pred << "\n";
    }

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
    std::array<Losses, N> train(PINN& model,
        const torch::Tensor& physics_input,
        OptimType& optimizer,
        const OptimizerSupplements& optSupplements)
    {
        auto current_lr{ optSupplements.learning_rate };
        std::array<Losses, N> all_losses;

        for (int epoch = 0; epoch < optSupplements.epochs; ++epoch) {
            try {
                Beam::Losses losses{};

                // Closure function: performs zero_grad, forward pass, backward pass, and returns loss.
                auto closure = [&]() -> torch::Tensor {
                    optimizer.zero_grad();

                    // Forward pass (loss computation must be inside closure for LBFGS).
                    losses = Losses::compute_losses(model, physics_input);
                    auto loss = losses.get_total_loss();

                    // NaN/Inf detection and adaptive learning rate.
                    if (torch::isnan(loss).any().item<bool>() || torch::isinf(loss).any().item<bool>()) {
                        std::cerr << "Optimizer Warning: Loss is NaN or Inf in epoch " << epoch << std::endl;
                        current_lr *= 0.5f;
                        std::cerr << "Adjusting learning rate to " << current_lr << std::endl;
                        for (auto& param_group : optimizer.param_groups()) {
                            auto& options = static_cast<OptimOptionType&>(param_group.options());
                            options.lr(current_lr);
                        }
                        return torch::tensor(1.0f, torch::requires_grad(true));  // Dummy loss
                    }

                    // Backward pass with optional graph retention.
                    loss.backward({}, Global::keep_graph);
                    return loss;
                    };

                // Optionally run closure before optimizer.step() if requested.
                if (optSupplements.call_closure) {
                    auto loss = closure();
                    (void)loss; // Suppress unused variable warning if not used.
                }

                // Gradient clipping to avoid exploding gradients.
                torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);

                // Step optimizer (closure always passed for LBFGS compatibility).
                optimizer.step(closure);

                // Periodic loss logging.
                if (epoch % optSupplements.epochs_diff == 0) {
                    std::cout << "Epoch " << epoch << "/" << optSupplements.epochs
                        << " - Total: " << losses.get_total_loss().item<float>()
                        << " | Physics: " << losses.get_physics_loss().item<float>()
                        << " | Boundary: " << losses.get_boundary_loss().item<float>()
                        << " | L2 term: " << losses.get_regularisation_loss().item<float>()
                        << " - Learning rate: " << current_lr << std::endl;
                    all_losses[epoch / optSupplements.epochs_diff] = std::move(losses);
                }
            }
            catch (const std::exception& e) {
                std::cout << "Error during Optimizer epoch " << epoch << ": " << e.what() << std::endl;
            }
        }

        return all_losses;
    }
}

/*
Vorschläge und Anmerkungen (keine Code-Änderung!):
Hardcoded loss weights (+2.0f * boundary etc.) könnten als Konstante oder Parameter geführt werden.

Die Verwendung von Global::keep_graph ist etwas "magisch" – eventuell explizit dokumentieren oder kapseln.

In physics_loss: Bei Problemen mit Gradientenstabilität wäre ein optionaler Check der .grad_fn() hilfreich.

std::pair<torch::Tensor, torch::Tensor> in generate_training_data() enthält ein Dummy-Ziel. Vielleicht besser std::tuple<torch::Tensor> oder ein spezieller Struct.

Fehlerbehandlung (try-catch) ist gut — aber torch::tensor(0.0f, ...) ohne Device-Spezifikation kann zu Problemen führen (nur wenn du CUDA verwendest).



*/