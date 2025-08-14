#pragma once

#include <memory>
#include <torch/torch.h>
#include "Global.h"
#include "Pinn.h"

namespace Beam {

    /**
     * @brief Struct to hold different loss components.
     */
    class Losses {
    public:
        Losses() = default;
        Losses(PINN& model, const torch::Tensor& input, const float lambda_reg = 0.0f) : model_pr{ std::make_unique<PINN>(model) } {
            physics_loss(input);
            boundary_loss();
            compute_l2_regularization(lambda_reg);
            total = physics + 2.0f * boundary + lambda_reg * l2_reg;
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
    static Losses compute_losses(PINN& model, const torch::Tensor& physics_input) {
        try {
            Losses losses(model, physics_input);
            return losses;
        }
        catch (const std::exception& e) {
            return Losses{};
        }
    }

    torch::Tensor get_total_loss() const {
        return total;
    }

    torch::Tensor get_physics_loss() const {
        return physics;
    }

    torch::Tensor get_boundary_loss() const {
        return boundary;
    }

    torch::Tensor get_regularisation_loss() const {
        return l2_reg;
    }

    private:
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
         */
        void physics_loss(const torch::Tensor& input, float EI = 1.0f) {
            try {
                auto x = input.clone().requires_grad_(true);
                auto u = model_pr->forward(x);
                auto ones = torch::ones_like(u);

                auto du_dx = torch::autograd::grad({ u }, { x }, { ones }, Global::keep_graph, true)[0];
                auto d2u_dx2 = torch::autograd::grad({ du_dx }, { x }, { ones }, Global::keep_graph, true)[0];
                auto d3u_dx3 = torch::autograd::grad({ d2u_dx2 }, { x }, { ones }, Global::keep_graph, true)[0];
                auto d4u_dx4 = torch::autograd::grad({ d3u_dx3 }, { x }, { ones }, Global::keep_graph, true)[0];

                auto q = torch::ones_like(x);  // Constant load q(x) = 1

                auto residual = EI * d4u_dx4 - q;

                physics = torch::mean(residual.pow(2));
            }
            catch (const std::exception& e) {
                std::cerr << "Error in physics_loss: " << e.what() << std::endl;
                physics = torch::tensor(0.0f, torch::requires_grad(true));
            }
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
        void boundary_loss() {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

            // x = 0 point
            auto x0 = torch::zeros({ 1, 1 }, options).set_requires_grad(true);
            auto u0 = model_pr->forward(x0);
            auto du0 = torch::autograd::grad({ u0 }, { x0 }, { torch::ones_like(u0) },
                Global::keep_graph, true)[0];

            // x = 1 point
            auto x1 = torch::ones({ 1, 1 }, options).set_requires_grad(true);
            auto u1 = model_pr->forward(x1);

            auto du1 = torch::autograd::grad({ u1 }, { x1 }, { torch::ones_like(u1) },
                Global::keep_graph, true)[0];
            auto d2u1 = torch::autograd::grad({ du1 }, { x1 }, { torch::ones_like(du1) },
                Global::keep_graph, true)[0];
            auto d3u1 = torch::autograd::grad({ d2u1 }, { x1 }, { torch::ones_like(d2u1) },
                Global::keep_graph, true)[0];

            // Sum of MSE losses for each boundary condition
            boundary = torch::mse_loss(u0, torch::zeros_like(u0)) +
                torch::mse_loss(du0, torch::zeros_like(du0)) +
                torch::mse_loss(d2u1, torch::zeros_like(d2u1)) +
                torch::mse_loss(d3u1, torch::zeros_like(d3u1));

        }

        /**
         * @brief Computes L2 (Tikhonov) regularization loss on all model parameters.
         *
         * @param model The PINN model
         * @param lambda_reg Regularization strength (0 = no regularization)
         * @return Scalar L2 regularization loss
         */
        void compute_l2_regularization(const float lambda_reg) {
            if (lambda_reg <= 0.0f) {
                l2_reg = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat32));
            }

            torch::Tensor l2 = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat32));
            for (const auto& param : model_pr->parameters()) {
                l2 += torch::norm(param, 2).pow(2);
            }

            l2_reg = lambda_reg * l2;
        }

        std::unique_ptr<PINN> model_pr;
        torch::Tensor total;            // Sum of all losses
        torch::Tensor physics;          // Loss of deviations from the physical law F(u) = 0
        torch::Tensor boundary;         // Loss of deviations from the boundary conditions on u
        torch::Tensor l2_reg;           // Penalty term, if Tikhonov L2 regularisation is active
    };
}
