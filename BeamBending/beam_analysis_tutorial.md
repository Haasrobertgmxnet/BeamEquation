# Cantilever Beam Analysis: PINN vs FEM vs Analytical Solution
# Kragträger-Analyse: PINN vs FEM vs Analytische Lösung

This notebook compares three different approaches for solving the cantilever beam deflection problem under uniform load:
- **PINN** (Physics-Informed Neural Networks) with ADAM and L-BFGS optimizers
- **FEM** (Finite Element Method)
- **Analytical Solution**

Dieses Notebook vergleicht drei verschiedene Ansätze zur Lösung des Kragträger-Durchbiegungsproblems unter gleichmäßiger Last:
- **PINN** (Physics-Informed Neural Networks) mit ADAM und L-BFGS Optimierern
- **FEM** (Finite-Elemente-Methode)
- **Analytische Lösung**

## Import Libraries / Bibliotheken importieren

```python
import time
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

# Import custom modules for PINN and FEM calculations
# Importiere benutzerdefinierte Module für PINN und FEM Berechnungen
from Beam_PINN import calculate_pinn
from Beam_FEM import calculate_fem
```

## Utility Functions / Hilfsfunktionen

### Random Seed Setting / Zufallszahlen-Seed setzen

This function ensures reproducibility by setting seeds for all random number generators used in the analysis.

Diese Funktion gewährleistet Reproduzierbarkeit durch das Setzen von Seeds für alle verwendeten Zufallszahlengeneratoren.

```python
def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility across PyTorch, NumPy, and Python.
    Setze den Zufallszahlen-Seed für Reproduzierbarkeit in PyTorch, NumPy und Python.

    Args:
        seed (int): Random seed value. / Zufallszahlen-Seed Wert.
    """
    # Set PyTorch random seed / PyTorch Zufallszahlen-Seed setzen
    torch.manual_seed(seed)
    # Set NumPy random seed / NumPy Zufallszahlen-Seed setzen
    np.random.seed(seed)
    # Set Python random seed / Python Zufallszahlen-Seed setzen
    random.seed(seed)
    
    # Set CUDA random seed if available / CUDA Zufallszahlen-Seed setzen falls verfügbar
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior / Deterministisches Verhalten sicherstellen
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Analytical Solution / Analytische Lösung

The exact analytical solution for a cantilever beam under uniform load, expressed in dimensionless form.

Die exakte analytische Lösung für einen Kragträger unter gleichmäßiger Last, ausgedrückt in dimensionsloser Form.

```python
def exact_solution(z: torch.Tensor) -> torch.Tensor:
    """
    Analytical solution to the dimensionless cantilever beam deflection problem.
    Analytische Lösung des dimensionslosen Kragträger-Durchbiegungsproblems.

    Args:
        z (torch.Tensor): Dimensionless spatial coordinate (0 to 1).
                         Dimensionslose Raumkoordinate (0 bis 1).

    Returns:
        torch.Tensor: Dimensionless deflection at point z.
                     Dimensionslose Durchbiegung am Punkt z.
    """
    # Fourth-order polynomial solution for uniform load
    # Polynomlösung vierter Ordnung für gleichmäßige Last
    return (1 / 24) * z ** 2 * (z ** 2 - 4 * z + 6)
```

### Error Calculation / Fehlerberechnung

Function to compute Root Mean Square Error (RMSE) and relative L2 error between predictions and reference values.

Funktion zur Berechnung des Root Mean Square Error (RMSE) und des relativen L2-Fehlers zwischen Vorhersagen und Referenzwerten.

```python
def calc_errors(x: np.ndarray, x_ref: np.ndarray) -> Tuple[float, float]:
    """
    Calculate RMSE and relative L2 error between predictions and reference.
    Berechne RMSE und relativen L2-Fehler zwischen Vorhersagen und Referenz.

    Args:
        x (np.ndarray): Predicted values. / Vorhergesagte Werte.
        x_ref (np.ndarray): Reference (exact or FEM) values. / Referenzwerte (exakt oder FEM).

    Returns:
        Tuple[float, float]: (RMSE, Relative L2 error) / (RMSE, Relativer L2-Fehler)
    """
    # Calculate Root Mean Square Error / Root Mean Square Error berechnen
    rmse = np.sqrt(np.mean((x - x_ref) ** 2))
    
    # Calculate relative L2 norm error / Relativen L2-Norm-Fehler berechnen
    rel = np.linalg.norm(x - x_ref, 2) / np.linalg.norm(x_ref, 2)
    
    return rmse, rel
```

## Main Analysis Function / Hauptanalysefunktion

This is the core function that orchestrates the entire comparison study between PINN, FEM, and analytical solutions.

Dies ist die Kernfunktion, die die gesamte Vergleichsstudie zwischen PINN, FEM und analytischen Lösungen koordiniert.

```python
def main() -> None:
    """
    Main execution function that compares PINN (ADAM + L-BFGS), FEM, and exact solution
    for a cantilever beam under uniform load. Includes training, evaluation, error analysis,
    and visualization.
    
    Hauptausführungsfunktion, die PINN (ADAM + L-BFGS), FEM und exakte Lösung
    für einen Kragträger unter gleichmäßiger Last vergleicht. Beinhaltet Training, 
    Auswertung, Fehleranalyse und Visualisierung.
    """
    # Set random seed for reproducibility / Zufallszahlen-Seed für Reproduzierbarkeit setzen
    set_seed(35)
```

### Physical Parameters / Physikalische Parameter

Definition of the cantilever beam's physical properties and loading conditions.

Definition der physikalischen Eigenschaften und Belastungsbedingungen des Kragträgers.

```python
    # Beam parameters / Trägerparameter
    L: float = 5.0             # Length of the beam [m] / Länge des Trägers [m]
    E: float = 210e9           # Elastic modulus [Pa] / Elastizitätsmodul [Pa]
    I: float = 1e-6            # Area moment of inertia [m^4] / Flächenträgheitsmoment [m^4]
    q: float = 1000.0          # Uniform load [N/m] / Gleichmäßige Last [N/m]

    # Sampling parameters / Sampling-Parameter
    n_samples: int = 100       # Number of evaluation/collocation points / Anzahl der Auswertungs-/Kollokationspunkte

    # Scaling factors for dimensionless form / Skalierungsfaktoren für dimensionslose Form
    w_char: float = q * L**4 / (E * I)        # Characteristic deflection / Charakteristische Durchbiegung
    scaling_theta: float = w_char / L          # Scaling for slope / Skalierung für Neigung
```

### PINN Training / PINN Training

Execute PINN training using both ADAM and L-BFGS optimizers.

PINN-Training mit sowohl ADAM als auch L-BFGS Optimierern ausführen.

```python
    # PINN training / PINN Training
    start = time.time()
    # Train neural network with ADAM followed by L-BFGS optimization
    # Trainiere neuronales Netz mit ADAM gefolgt von L-BFGS Optimierung
    adam_results, time_adam, lbfgs_results, time_lbfgs = calculate_pinn(
        adam_epochs=200,           # Number of ADAM training epochs / Anzahl ADAM Trainingsepochen
        lbfgs_epochs=300,          # Number of L-BFGS training epochs / Anzahl L-BFGS Trainingsepochen
        n_points=n_samples,        # Number of collocation points / Anzahl Kollokationspunkte
        loss_threshold=1e-8        # Convergence threshold / Konvergenzschwelle
    )
    time_pinn = time.time() - start
```

### FEM Computation / FEM Berechnung

Compute the finite element solution for comparison.

Finite-Elemente-Lösung zur Vergleichsberechnung.

```python
    # FEM computation / FEM Berechnung
    start = time.time()
    # Calculate deflection and slope using finite element method
    # Berechne Durchbiegung und Neigung mit der Finite-Elemente-Methode
    v_fem, v_fem_theta = calculate_fem(n_samples)
    time_fem = time.time() - start
```

### Performance Logging / Leistungsprotokollierung

Log the computational times for each method.

Protokollierung der Berechnungszeiten für jede Methode.

```python
    # Log runtimes / Laufzeiten protokollieren
    print(f"\nExecution Times:")
    print(f"PINN Total:     {time_pinn:.4f} s")
    print(f"  ADAM:         {time_adam:.4f} s")
    print(f"  LBFGS:        {time_lbfgs:.4f} s")
    print(f"FEM:            {time_fem:.4f} s\n")
```

### Results Extraction / Ergebnisextraktion

Extract and process results from different methods.

Ergebnisse von verschiedenen Methoden extrahieren und verarbeiten.

```python
    # Extract results / Ergebnisse extrahieren
    z_eval, v_adam, loss_history = adam_results         # ADAM results / ADAM Ergebnisse
    _, v_lbfgs, _ = lbfgs_results                       # L-BFGS results / L-BFGS Ergebnisse
    
    # Calculate exact solution at evaluation points / Exakte Lösung an Auswertungspunkten berechnen
    v_exact: np.ndarray = exact_solution(z_eval).detach().numpy().flatten()
```

### Dimensional Scaling / Dimensionale Skalierung

Convert dimensionless results back to physical units for comparison and visualization.

Dimensionslose Ergebnisse zurück in physikalische Einheiten für Vergleich und Visualisierung umwandeln.

```python
    # Scale dimensionless displacements to real-world units
    # Skaliere dimensionslose Verschiebungen zu realen Einheiten
    w_exact = v_exact * w_char                          # Exact solution in [m] / Exakte Lösung in [m]
    w_adam = v_adam * w_char                            # ADAM solution in [m] / ADAM Lösung in [m]
    w_lbfgs = v_lbfgs * w_char                          # L-BFGS solution in [m] / L-BFGS Lösung in [m]
    w_fem = v_fem * w_char                              # FEM solution in [m] / FEM Lösung in [m]
    w_fem_theta = v_fem_theta * scaling_theta           # FEM slope scaled / FEM Neigung skaliert
    x_eval = z_eval.numpy().flatten() * L               # Physical coordinates [m] / Physikalische Koordinaten [m]
```

### Error Analysis / Fehleranalyse

Comprehensive error analysis comparing all methods against each other.

Umfassende Fehleranalyse, die alle Methoden miteinander vergleicht.

```python
    # Compute errors / Fehler berechnen
    # Errors against exact solution / Fehler gegenüber exakter Lösung
    rmse_adam_exact, rel_adam_exact = calc_errors(w_adam, w_exact)
    rmse_lbfgs_exact, rel_lbfgs_exact = calc_errors(w_lbfgs, w_exact)
    rmse_fem_exact, rel_fem_exact = calc_errors(w_fem, w_exact)
    
    # Cross-method comparisons / Methodenübergreifende Vergleiche
    rmse_adam_fem, rel_adam_fem = calc_errors(w_adam, w_fem)
    rmse_lbfgs_fem, rel_lbfgs_fem = calc_errors(w_lbfgs, w_fem)
    rmse_adam_lbfgs, rel_adam_lbfgs = calc_errors(w_adam, w_lbfgs)

    # Print error summary / Fehlerzusammenfassung ausgeben
    print("RMSE Errors:")
    print(f"ADAM vs Exact:\t{rmse_adam_exact:.6e}")
    print(f"LBFGS vs Exact:\t{rmse_lbfgs_exact:.6e}")
    print(f"FEM vs Exact:\t{rmse_fem_exact:.6e}")
    print(f"ADAM vs FEM:\t{rmse_adam_fem:.6e}")
    print(f"LBFGS vs FEM:\t{rmse_lbfgs_fem:.6e}")
    print(f"ADAM vs LBFGS:\t{rmse_adam_lbfgs:.6e}\n")

    print("Relative L2 Errors:")
    print(f"ADAM vs Exact:\t{rel_adam_exact:.6e}")
    print(f"LBFGS vs Exact:\t{rel_lbfgs_exact:.6e}")
    print(f"FEM vs Exact:\t{rel_fem_exact:.6e}")
    print(f"ADAM vs FEM:\t{rel_adam_fem:.6e}")
    print(f"LBFGS vs FEM:\t{rel_lbfgs_fem:.6e}")
    print(f"ADAM vs LBFGS:\t{rel_adam_lbfgs:.6e}\n")
```

## Visualization / Visualisierung

### PINN vs Analytical Solution / PINN vs Analytische Lösung

Comparison of PINN methods (ADAM and L-BFGS) against the exact analytical solution.

Vergleich der PINN-Methoden (ADAM und L-BFGS) mit der exakten analytischen Lösung.

```python
    # --- Visualization / Visualisierung ---

    # Plot PINN vs Exact / PINN vs Exakt plotten
    plt.figure(figsize=(8, 5))
    # L-BFGS results (typically more accurate) / L-BFGS Ergebnisse (typischerweise genauer)
    plt.plot(x_eval, w_lbfgs, color='orange', label='LBFGS', linewidth=2, linestyle='-')
    # ADAM results / ADAM Ergebnisse
    plt.plot(x_eval, w_adam, color='red', label='ADAM', linewidth=2, linestyle='--')
    # Analytical reference solution / Analytische Referenzlösung
    plt.plot(x_eval, w_exact, color='navy', label='Analytical', linewidth=2, linestyle=':')
    plt.xlabel('x [m]')
    plt.ylabel('w(x) [m]')
    plt.title('Beam Deflection - PINN vs Exact')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
```

### PINN vs FEM Comparison / PINN vs FEM Vergleich

Comparison of PINN methods against the Finite Element Method solution.

Vergleich der PINN-Methoden mit der Finite-Elemente-Methoden-Lösung.

```python
    # Plot PINN vs FEM / PINN vs FEM plotten
    plt.figure(figsize=(8, 5))
    # L-BFGS results / L-BFGS Ergebnisse
    plt.plot(x_eval, w_lbfgs, color='orange', label='LBFGS', linewidth=2, linestyle='-')
    # ADAM results / ADAM Ergebnisse
    plt.plot(x_eval, w_adam, color='red', label='ADAM', linewidth=2, linestyle='--')
    # FEM reference solution / FEM Referenzlösung
    plt.plot(x_eval, w_fem, color='navy', label='FEM', linewidth=2, linestyle=':')
    plt.xlabel('x [m]')
    plt.ylabel('w(x) [m]')
    plt.title('Beam Deflection - PINN vs FEM')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
```

### Training Loss History / Trainingsverlusthistorie

Visualization of the PINN training convergence during the ADAM optimization phase.

Visualisierung der PINN-Trainingskonvergenz während der ADAM-Optimierungsphase.

```python
    # Loss plot / Verlustplot
    if loss_history is not None:
        plt.figure()
        # Plot training loss on logarithmic scale / Trainingsverlust auf logarithmischer Skala plotten
        plt.plot(loss_history)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss (ADAM)")
        plt.grid(True)

    # Display all plots / Alle Plots anzeigen
    plt.show()
```

## Execution / Ausführung

```python
if __name__ == "__main__":
    main()
```

---

## Summary / Zusammenfassung

This notebook demonstrates a comprehensive comparison of three approaches to solving the cantilever beam deflection problem:

1. **Physics-Informed Neural Networks (PINN)** with both ADAM and L-BFGS optimizers
2. **Finite Element Method (FEM)**
3. **Analytical (exact) solution**

The analysis includes performance timing, error metrics (RMSE and relative L2 error), and visualization of results.

Dieses Notebook demonstriert einen umfassenden Vergleich von drei Ansätzen zur Lösung des Kragträger-Durchbiegungsproblems:

1. **Physics-Informed Neural Networks (PINN)** mit sowohl ADAM als auch L-BFGS Optimierern
2. **Finite-Elemente-Methode (FEM)**
3. **Analytische (exakte) Lösung**

Die Analyse umfasst Leistungsmessungen, Fehlermetriken (RMSE und relativer L2-Fehler) und Visualisierung der Ergebnisse.