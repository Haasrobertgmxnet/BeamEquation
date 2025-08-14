#pragma once

#include <torch/torch.h>
#include "Pinn.h"
#include "Global.h"

namespace Beam {

    /**
     * @brief Struct to hold different loss components.
     */
    struct Losses {
        //PINN& model;
        torch::Tensor total;        // Sum of all losses
        torch::Tensor physics;      // Loss of deviations from the physical law F(u) = 0
        torch::Tensor boundary;     // Loss of deviations from the boundary conditions on u
        torch::Tensor l2_reg;       // Penalty term, if Tikhonov L2 regularisation is active
    };

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


    // Add the calculation methods for the losses to the struct Losses?
    /**
     * @brief Computes the boundary condition loss for the beam.
     *
     * Enforces:
     * - u(0) = 0
     * - u'(0) = 0
     * - u''(1) = 0
     * - u'''(1) = 0
     *
     * @param model The PINN model
     * @return Mean squared error of all boundary residuals
     */
    [[nodiscard]]
    torch::Tensor boundary_loss(PINN& model) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

        // x = 0 point
        auto x0 = torch::zeros({ 1, 1 }, options).set_requires_grad(true);
        auto u0 = model.forward(x0);
        auto du0 = torch::autograd::grad({ u0 }, { x0 }, { torch::ones_like(u0) },
            Global::keep_graph, true)[0];

        // x = 1 point
        auto x1 = torch::ones({ 1, 1 }, options).set_requires_grad(true);
        auto u1 = model.forward(x1);

        auto du1 = torch::autograd::grad({ u1 }, { x1 }, { torch::ones_like(u1) },
            Global::keep_graph, true)[0];
        auto d2u1 = torch::autograd::grad({ du1 }, { x1 }, { torch::ones_like(du1) },
            Global::keep_graph, true)[0];
        auto d3u1 = torch::autograd::grad({ d2u1 }, { x1 }, { torch::ones_like(d2u1) },
            Global::keep_graph, true)[0];

        // Sum of MSE losses for each boundary condition
        auto loss = torch::mse_loss(u0, torch::zeros_like(u0)) +
            torch::mse_loss(du0, torch::zeros_like(du0)) +
            torch::mse_loss(d2u1, torch::zeros_like(d2u1)) +
            torch::mse_loss(d3u1, torch::zeros_like(d3u1));

        return loss;
    }

    /**
     * @brief Computes the physics loss (residual of the Euler-Bernoulli beam equation).
     *
     * The differential equation:
     *     EI * d4u/dx4 = q(x)
     * where q(x) = 1 is used here.
     *
     * @param model The PINN model
     * @param input Sample input points (x in R^{N×1})
     * @param EI Bending stiffness coefficient (default = 1.0)
     * @return MSE of the PDE residuals
     */
    [[nodiscard]]
    torch::Tensor physics_loss(PINN& model, torch::Tensor input, float EI = 1.0f) {
        try {
            auto x = input.clone().requires_grad_(true);
            auto u = model.forward(x);
            auto ones = torch::ones_like(u);

            auto du_dx = torch::autograd::grad({ u }, { x }, { ones }, Global::keep_graph, true)[0];
            auto d2u_dx2 = torch::autograd::grad({ du_dx }, { x }, { ones }, Global::keep_graph, true)[0];
            auto d3u_dx3 = torch::autograd::grad({ d2u_dx2 }, { x }, { ones }, Global::keep_graph, true)[0];
            auto d4u_dx4 = torch::autograd::grad({ d3u_dx3 }, { x }, { ones }, Global::keep_graph, true)[0];

            auto q = torch::ones_like(x);  // Constant load q(x) = 1

            auto residual = EI * d4u_dx4 - q;

            return torch::mean(residual.pow(2));
        }
        catch (const std::exception& e) {
            std::cerr << "Error in physics_loss: " << e.what() << std::endl;
            return torch::tensor(0.0f, torch::requires_grad(true));
        }
    }

    /**
     * @brief Computes L2 (Tikhonov) regularization loss on all model parameters.
     *
     * @param model The PINN model
     * @param lambda_reg Regularization strength (0 = no regularization)
     * @return Scalar L2 regularization loss
     */
    [[nodiscard]]
    torch::Tensor compute_l2_regularization(PINN& model, const float lambda_reg) {
        if (lambda_reg <= 0.0f) {
            return torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat32));
        }

        torch::Tensor l2 = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat32));
        for (const auto& param : model.parameters()) {
            l2 += torch::norm(param, 2).pow(2);
        }

        return lambda_reg * l2;
    }

    /**
     * @brief Computes the total loss and its components (physics, boundary, L2).
     *
     * Total loss = physics + 2 × boundary + 1 × L2 (hardcoded weights)
     *
     * @param model The PINN model
     * @param physics_input Input points for physics loss
     * @return Struct containing all loss terms
     */
    [[nodiscard]]
    Losses compute_losses(PINN& model, const torch::Tensor& physics_input) {
        try {
            auto loss_physics = physics_loss(model, physics_input);
            auto loss_boundary = boundary_loss(model);
            auto l2_reg = compute_l2_regularization(model, 0.0f);

            auto total = loss_physics + 2.0f * loss_boundary + l2_reg;

            return { total, loss_physics, loss_boundary, l2_reg };
        }
        catch (const std::exception& e) {
            std::cerr << "Error in compute_losses: " << e.what() << std::endl;
            auto dummy = torch::tensor(1.0f, torch::requires_grad(true));
            return { dummy, dummy, dummy, dummy };
        }
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
                    losses = compute_losses(model, physics_input);
                    auto loss = losses.total;

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
}

/*
Vorschläge und Anmerkungen (keine Code-Änderung!):
Hardcoded loss weights (+2.0f * boundary etc.) könnten als Konstante oder Parameter geführt werden.

Die Verwendung von Global::keep_graph ist etwas "magisch" – eventuell explizit dokumentieren oder kapseln.

In physics_loss: Bei Problemen mit Gradientenstabilität wäre ein optionaler Check der .grad_fn() hilfreich.

std::pair<torch::Tensor, torch::Tensor> in generate_training_data() enthält ein Dummy-Ziel. Vielleicht besser std::tuple<torch::Tensor> oder ein spezieller Struct.

Fehlerbehandlung (try-catch) ist gut — aber torch::tensor(0.0f, ...) ohne Device-Spezifikation kann zu Problemen führen (nur wenn du CUDA verwendest).



*/