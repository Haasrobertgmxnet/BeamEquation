# BeamEquation – Praxis-Tutorial (DE)

**Stand:** 2025-08-27  
**Repository:** https://github.com/Haasrobertgmxnet/BeamEquation.git

Dieses Tutorial begleitet dich Schritt für Schritt durch das Repo und zeigt, wie du
1) das **Forward-Problem** (Durchbiegung) mit **FEM** und **PINN** löst und
2) das **Inverse-Problem** (Schätzung des Youngschen Moduls *E*) mit einem **PINN** angehst.

---

## 1. Installation & Start

```bash
git clone https://github.com/Haasrobertgmxnet/BeamEquation.git
cd BeamEquation
# optional: python -m venv .venv && source .venv/bin/activate
pip install -r BeamBending/requirements.txt  # falls vorhanden
```

Tipp: Für GPU-Training installiere eine passende PyTorch-Version (siehe PyTorch-Website).

---

## 2. Repo-Überblick

- **Beam.Python/** – Python-Implementierungen für **FEM** und **PINN** (Forward).  
  - `Beam.Python.py` – Einstiegsskript/Pipeline  
  - `Beam_FEM.py` – FEM-Helfer  
  - `Beam_PINN.py` – PINN-Helfer  
- **Parameteridentification.Python/** – PINN für **Inverse-Problem** (E-Schätzung).  
  - `Parameteridentification.Python.py` – Einstiegsskript  
  - `Beam_PINN_inverse.py` – PINN-Definition für inverse Aufgaben  
- **BeamBending/** – Enthält `Documentation.pdf` mit sämtlichen Hintergründen, Formeln und Resultaten.

---

## 3. Schnellstart (Forward)

### 3.1 FEM-Referenz

```python
from Beam_FEM import BeamModel, PointLoad, Supports

L, E, I = 1.0, 2.1e11, 4.76e-10
model = BeamModel(L=L, n_el=200, E=E, I=I)
model.add_support(Supports.Fixed(0.0))       # Einspannung
model.add_load(PointLoad(x=L, Fy=-100.0))    # Endlast

res = model.solve()
res.plot_all()  # V(x), M(x), w(x)
```

### 3.2 PINN-Training

```python
from Beam_PINN import BeamPINN, PINNConfig

cfg = PINNConfig(
    n_hidden=4, n_neurons=64, lr=1e-3, epochs=5000,
    collocation_points=2000, bc_weight=100.0
)
pinn = BeamPINN(L=1.0, E=E, I=I, config=cfg)
history = pinn.train()

# Vergleich
x = res['x']; w_fem = res['w']; w_pinn = pinn.predict(x)
```

---

## 4. Inverses Problem (E-Schätzung)

```python
from Beam_PINN_inverse import InversePINN, InverseConfig

cfg = InverseConfig(n_hidden=4, n_neurons=64, lr=5e-4, epochs=8000, data_weight=50.0)
inv = InversePINN(L=1.0, I=4.76e-10, E_init=1.0e11, config=cfg)
inv.load_measurements('measured_deflection.npz')  # (x, w)-Daten
inv.train()
print('Geschätztes E:', inv.current_E())
```

**Messdaten** kannst du entweder aus deiner FEM-Lösung synthetisch erzeugen oder aus realen Messungen laden.

---

## 5. Reproduzierbarkeit & Tipps

- Setze Seeds für `numpy`, `random`, `torch`.
- Dokumentiere Hyperparameter (Netz, Lernrate, Gewichte).
- Achte auf Batch-Normalisierung/Regularisierung nur wenn sinnvoll.
- Prüfe PINN-Residuals und **Boundary Condition**-Erfüllung.
- Verwende genügend **Kollokationspunkte**; erhöhe ggf. `bc_weight`.

---

## 6. Notebooks

- `00_Intro_and_Setup.ipynb` – Überblick & Setup
- `01_Forward_PINN_vs_FEM.ipynb` – Forward: PINN vs FEM
- `02_Inverse_Youngs_Modulus_PINN.ipynb` – Invers: E-Schätzung

> Öffne die Notebooks in Jupyter Lab/VS Code und führe die Zellen **lokal** aus.

Viel Erfolg!
