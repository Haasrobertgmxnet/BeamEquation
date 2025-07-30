#define CATCH_CONFIG_MAIN
#include <torch/torch.h>
#include <catch2/catch_all.hpp>  // Catch2 v3 header
#include "Beam.h"

/// \brief Tests the forward pass of the PINN model on random input.
/// \details Ensures the output has correct shape and no runtime errors occur.
TEST_CASE("PINN: basic forward pass") {
    Beam::PINN model;

    // Generate a small batch of input data
    auto input = torch::rand({ 5, 1 }, torch::requires_grad(true));

    // Forward pass through the model
    auto output = model.forward(input);

    // Check output shape matches expected dimensions (batch size x 1)
    REQUIRE(output.sizes() == torch::IntArrayRef({ 5, 1 }));
}

/// \brief Tests that the physics loss can be computed without throwing exceptions.
/// \details Checks that the output is finite (no NaNs/Infs).
TEST_CASE("Physics loss: no crash on valid input") {
    Beam::PINN model;

    // Random input data in domain [0,1]
    auto input = torch::rand({ 10, 1 }, torch::requires_grad(true));

    // Compute physics-informed loss (ODE residual)
    auto loss = Beam::physics_loss(model, input);

    // Validate result: must be a finite scalar
    REQUIRE(torch::isfinite(loss).all().item<bool>());
}

/// \brief Tests that boundary condition loss does not crash.
/// \details Verifies that the boundary loss is finite and computable.
TEST_CASE("Boundary loss: no crash") {
    Beam::PINN model;

    // Compute boundary condition residual
    auto loss = Beam::boundary_loss(model);

    // Should yield finite result (no NaN/Inf)
    REQUIRE(torch::isfinite(loss).all().item<bool>());
}
