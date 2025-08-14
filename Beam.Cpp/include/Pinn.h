#pragma once
#include <torch/torch.h>

namespace Beam {

    /**
     * @brief A Physics-Informed Neural Network (PINN) model for 1D Euler-Bernoulli beam deflection.
     *
     * Network architecture:
     *     Input: x in R
     *     Hidden: two fully-connected layers with tanh activation
     *     Output: u(x) in R
     */
    class PINN : public torch::nn::Module {
    public:
        PINN() {
            // Network structure: 1 input -> 5 neurons -> 5 neurons -> 1 output
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
        [[nodiscard]]
        torch::Tensor forward(torch::Tensor x) {
            x = torch::tanh(fc1->forward(x));
            x = torch::tanh(fc2->forward(x));
            x = fc3->forward(x);
            return x;
        }

    private:
        torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
    };

    
}
