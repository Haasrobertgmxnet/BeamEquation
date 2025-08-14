#pragma once

#include <torch/torch.h>
#include "Pinn.h"
#include "Global.h"
#include "Losses.h"

namespace Beam {

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
    std::array<Losses, N> train(
        PINN& model,
        const torch::Tensor& physics_input,
        OptimType& optimizer,
        const OptimizerSupplements& optSupplements,
        int patience = 20,                // Early Stopping patience
        float min_delta = 1e-5f,          // Minimal improvement
        int val_points = 200              // Points for validation
    ) {
        auto current_lr{ optSupplements.learning_rate };
        std::array<Losses, N> all_losses;

        float best_val_loss = std::numeric_limits<float>::infinity();
        int epochs_no_improve = 0;

        // Buffer for best model state (in RAM)
        std::stringstream best_model_buffer;

        for (int epoch = 0; epoch < optSupplements.epochs; ++epoch) {
            try {
                Beam::Losses losses{};

                auto closure = [&]() -> torch::Tensor {
                    optimizer.zero_grad();
                    losses = Losses::compute_losses(model, physics_input);
                    auto loss = losses.get_total_loss();

                    if (torch::isnan(loss).any().item<bool>() || torch::isinf(loss).any().item<bool>()) {
                        std::cerr << "Optimizer Warning: Loss is NaN/Inf in epoch " << epoch << std::endl;
                        current_lr *= 0.5f;
                        for (auto& param_group : optimizer.param_groups()) {
                            auto& options = static_cast<OptimOptionType&>(param_group.options());
                            options.lr(current_lr);
                        }
                        return torch::tensor(1.0f, torch::requires_grad(true));
                    }

                    loss.backward({}, Global::keep_graph);
                    return loss;
                    };

                if (optSupplements.call_closure) {
                    closure();
                }

                torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);
                optimizer.step(closure);

                // === Early Stopping on new validation points ===
                {
                    auto val_input = Beam::generate_training_data(val_points);

                    model.eval();
                    auto val_losses = Losses::compute_losses(model, val_input);
                    float val_loss_value = val_losses.get_total_loss().item<float>();
                    model.train();

                    if (val_loss_value < best_val_loss - min_delta) {
                        best_val_loss = val_loss_value;
                        epochs_no_improve = 0;

                        // Save model state in RAM
                        best_model_buffer.str("");
                        best_model_buffer.clear();
                        torch::serialize::OutputArchive tmp;
                        model.save(tmp);
                        tmp.save_to(best_model_buffer);
                    }
                    else {
                        epochs_no_improve++;
                    }

                    if (epochs_no_improve >= patience) {
                        std::cout << "Early stopping triggered at epoch " << epoch << std::endl;

                        // Load best model state from RAM
                        torch::serialize::InputArchive ia;
                        ia.load_from(best_model_buffer);
                        model.load(ia);
                        break;
                    }
                }

                // Logging
                if (epoch % optSupplements.epochs_diff == 0) {
                    std::cout << "Epoch " << epoch << "/" << optSupplements.epochs
                        << " - Total: " << losses.get_total_loss().item<float>()
                        << " - Best Val Loss: " << best_val_loss
                        << " - LR: " << current_lr << std::endl;
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