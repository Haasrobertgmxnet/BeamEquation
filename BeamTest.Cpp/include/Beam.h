#pragma once

#include <torch/torch.h>
#include "Global.h"

namespace Beam {

    /**
     * @brief A Physics-Informed Neural Network (PINN) model for 1D Euler-Bernoulli beam deflection.
     *
     * Network architecture:
     *     Input: x ∈ ℝ¹
     *     Hidden: two fully-connected layers with tanh activation
     *     Output: u(x) ∈ ℝ¹
     */
    class PINN : public torch::nn::Module {
    public:
        PINN() {
            // Network structure: 1 input → 5 neurons → 5 neurons → 1 output
            fc1 = register_module("fc1", torch::nn::Linear(1, 5));
            fc2 = register_module("fc2", torch::nn::Linear(5, 5));
            fc3 = register_module("fc3", torch::nn::Linear(5, 1));

            // Initialize weights using Xavier initialization
            torch::nn::init::xavier_uniform_(fc1->weight);
            torch::nn::init::xavier_uniform_(fc2->weight);
            torch::nn::init::xavier_uniform_(fc3->weight);
        }

        /**
         * @brief Forward pass of the PINN.
         * @param x Input tensor of shape (N, 1)
         * @return Network output tensor of shape (N, 1)
         */
        torch::Tensor forward(torch::Tensor x) {
            x = torch::tanh(fc1->forward(x));
            x = torch::tanh(fc2->forward(x));
            x = fc3->forward(x);
            return x;
        }

    private:
        torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
    };

    /**
     * @brief Struct to hold different loss components.
     */
    struct Losses {
        torch::Tensor total;
        torch::Tensor physics;
        torch::Tensor boundary;
        torch::Tensor l2_reg;
    };

    // Number of epochs for training with Adam optimizer
    constexpr auto adam_epochs = uint16_t{ 500 };
    constexpr auto adam_epochs_diff = uint16_t{ 100 };

    /**
     * @brief Generates random 1D training points in the interval [0, 1].
     * @param n_points Number of training samples
     * @return Pair of tensors: input x ∈ ℝ^{n×1}, dummy targets y ∈ ℝ^{n×1}
     */
    std::pair<torch::Tensor, torch::Tensor> generate_training_data(const int n_points = 100) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

        auto x = torch::rand({ n_points, 1 }, options);  // Random points in [0, 1]
        auto y = torch::zeros_like(x);                   // Dummy target, not used in PINN directly

        return { x, y };
    }

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
     *     EI * d⁴u/dx⁴ = q(x)
     * where q(x) = 1 is used here.
     *
     * @param model The PINN model
     * @param input Sample input points (x ∈ ℝ^{N×1})
     * @param EI Bending stiffness coefficient (default = 1.0)
     * @return MSE of the PDE residuals
     */
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
    Losses compute_losses(PINN& model, const torch::Tensor& physics_input) {
        try {
            auto loss_physics = physics_loss(model, physics_input);
            auto loss_boundary = boundary_loss(model);
            auto l2_reg = compute_l2_regularization(model, 0.0f);

            auto total = loss_physics + 2.0f * loss_boundary + 1.0f * l2_reg;

            return { total, loss_physics, loss_boundary, l2_reg };
        }
        catch (const std::exception& e) {
            std::cerr << "Error in compute_losses: " << e.what() << std::endl;
            auto dummy = torch::tensor(1.0f, torch::requires_grad(true));
            return { dummy, dummy, dummy, dummy };
        }
    }
}
