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

namespace Model = Beam;

// Loesung visualisieren (einfache Konsolen-Ausgabe)
void visualize_solution(Model::PINN& model, int grid_size = 20) {
    model.eval();
    torch::NoGradGuard no_grad;

    torch::Device device = torch::kCPU;
    for (const auto& p : model.parameters()) {
        device = p.device(); break;
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    std::cout << "\nLösung der Euler-Bernoulli-Gleichung auf [0, 1]:\n";

    for (int i = 0; i < grid_size; ++i) {
        float x_val = static_cast<float>(i) / (grid_size - 1);  // x in [0, 1]

        auto x_tensor = torch::tensor({ {x_val} }, options);  // Shape: [1,1]

        torch::Tensor u_pred;
        try {

            u_pred = model.forward(x_tensor);
        }
        catch (const c10::Error& e) {
            std::cerr << "Fehler beim Forward-Pass bei x = " << x_val << ": " << e.what() << "\n";
            continue;
        }

        float u_val = u_pred.detach().to(torch::kCPU).item<float>();
        std::cout << std::fixed << std::setprecision(2) << x_val << "\t" << std::setprecision(5) << u_val << "\n";
    }
}


int main() {
    Helper::Timer tim;
    std::cout << "Physics-Informed Neural Network fuer Beam-Gleichung\n";
    std::cout << "====================================================\n\n";

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA verfügbar - verwende GPU\n";
    }
    else {
        std::cout << "Verwende CPU\n";
    }

    auto model = std::make_unique<Model::PINN>();
    model->to(device);

    auto [physics_input, _] = Model::generate_training_data(100);

    physics_input = physics_input.to(device);

    float current_lr = 0.1f;

    auto start_time = std::chrono::steady_clock::now();

    // === PHASE 1: Adam Optimizer (grobes Training) ===

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(current_lr));



    std::array<Model::Losses, Model::adam_epochs / Model::adam_epochs_diff> all_losses;

    std::cout << "\n[Phase 1] Adam Training...\n";

    for (int epoch = 0; epoch < Model::adam_epochs; ++epoch) {
        try {
            optimizer.zero_grad();

            auto losses = compute_losses(*model, physics_input);

            // NaN- oder Inf-Check
            if (torch::isnan(losses.total).any().item<bool>() || torch::isinf(losses.total).any().item<bool>()) {
                std::cerr << "Adam Warnung: Loss ist NaN oder Inf in Epoche " << epoch << std::endl;

                // Lernrate halbieren und im Optimizer einstellen
                current_lr *= 0.5f;
                std::cerr << "Setze Lernrate auf " << current_lr << std::endl;
                for (auto& param_group : optimizer.param_groups()) {
                    auto& options = static_cast<torch::optim::AdamOptions&>(param_group.options());
                    options.lr(current_lr);
                }
                continue;  // diese Epoche überspringen
            }

            losses.total.backward({}, /*Global::keep_graph=*/Global::keep_graph);

            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            optimizer.step();

            if (epoch % 200 == 0) {
                all_losses[epoch / Model::adam_epochs_diff] = losses;
                std::cout << "Epoch " << epoch << "/" << Model::adam_epochs
                    << " - Total: " << losses.total.item<float>()
                    << " | Physics: " << losses.physics.item<float>()
                    << " | Boundary: " << losses.boundary.item<float>()
                    << " - LR: " << current_lr << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cout << "Fehler in Adam Epoche " << epoch << ": " << e.what() << std::endl;
        }
    }

    // === PHASE 2: LBFGS Optimizer (Feinabstimmung) ===
    std::cout << "\n[Phase 2] LBFGS Finetuning...\n";

    torch::optim::LBFGS lbfgs(model->parameters(),
        torch::optim::LBFGSOptions(1.0)
        .max_iter(20)
        .tolerance_grad(1e-7)
        .tolerance_change(1e-9)
        .history_size(100));

    int lbfgs_epochs = 50; // 200;
    current_lr *= 0.5f;

    for (int epoch = 0; epoch < lbfgs_epochs; ++epoch) {
        try {
            auto closure = [&]() -> torch::Tensor {
                lbfgs.zero_grad();

                auto loss = compute_losses(*model, physics_input).total;

                // NaN- oder Inf-Check
                if (torch::isnan(loss).any().item<bool>() || torch::isinf(loss).any().item<bool>()) {
                    std::cerr << "LBFGS Warnung: Loss ist NaN oder Inf in Epoche " << epoch << std::endl;

                    // Lernrate halbieren und im Optimizer einstellen
                    current_lr *= 0.5f;
                    std::cerr << "Setze Lernrate auf " << current_lr << std::endl;
                    for (auto& param_group : lbfgs.param_groups()) {
                        auto& options = static_cast<torch::optim::LBFGSOptions&>(param_group.options());
                        options.lr(current_lr);
                    }
                    return torch::tensor(1.0f, torch::requires_grad(true));
                }


                loss.backward({}, /*Global::keep_graph=*/Global::keep_graph);
                return loss;
                };

            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            torch::Tensor loss = lbfgs.step(closure);

            if (epoch % 10 == 0) {
                std::cout << "LBFGS Epoch " << epoch << "/" << lbfgs_epochs
                    << " - Loss: " << loss.item<float>() << " - Learning Rate: " << current_lr << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cout << "Fehler in LBFGS Epoche " << epoch << ": " << e.what() << std::endl;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\nTraining abgeschlossen in " << duration.count() << " Sekunden\n";

    model->eval();
    visualize_solution(*model);

    return 0;
}
