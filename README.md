# Non-Linear System Identification

Identificação de sistemas não-lineares usando três algoritmos de seleção de estrutura, com backend de computação em **Rust** (via PyO3) e interface/plots em **Python**.

## Algoritmos

| Algoritmo | Descrição |
|---|---|
| **SEMP** | Structure Selection via ERR with Model Pruning — seleção forward + eliminação backward |
| **FROLS** | Forward Regression Orthogonal Least Squares — Gram-Schmidt com critério de parada por ESR |
| **Gram-Schmidt** | Seleção de estrutura por número fixo de termos (n_theta) — compara modelo completo vs selecionado |

## Estrutura do projeto

```
├── data/                    # Datasets (.csv)
├── notebooks/               # Jupyter notebooks
│   ├── main.ipynb           # Benchmark completo (3 algoritmos × 5 datasets)
│   ├── SEMP.ipynb           # SEMP em todos os datasets
│   ├── FROLS.ipynb          # FROLS em todos os datasets
│   └── NLSI.ipynb           # Gram-Schmidt em todos os datasets
├── rust_backend/            # Código-fonte Rust (PyO3)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # Bindings PyO3 (módulo rust_nlsi)
│       ├── semp.rs          # Algoritmo SEMP
│       ├── frols.rs         # Algoritmo FROLS
│       ├── gram_schmidt.rs  # Algoritmo Gram-Schmidt
│       ├── sysid.rs         # data_matrix, candidate_matrix, combinações
│       └── linalg.rs        # Inversão de matriz, least squares
├── src/                    # Pacote Python (wrappers + plots)
│   ├── __init__.py
│   ├── semp.py              # Classe Semp
│   ├── frols.py             # Classe Frols
│   ├── gram_schmidt.py      # Classe GramSchmidt
│   └── plotting.py          # Funções de plot (matplotlib)
└── README.md
```

## Pré-requisitos

- Python 3.13+
- Rust (via [rustup](https://rustup.rs/))
- [maturin](https://github.com/PyO3/maturin): `pip install maturin`

## Como compilar e rodar

### 1. Compilar o backend Rust

```bash
cd rust_backend
python -m maturin build --release
```

### 2. Instalar o módulo compilado

```bash
python -m pip install target/wheels/rust_nlsi-0.1.0-cp313-cp313-win_amd64.whl --force-reinstall
```

### 3. Instalar dependências Python

```bash
pip install numpy pandas matplotlib jinja2
```

### 4. Rodar os notebooks

Abra os notebooks na pasta `notebooks/` no VS Code ou Jupyter:

- **`main.ipynb`** — roda os 3 algoritmos em todos os 5 datasets e gera tabela comparativa
- **`SEMP.ipynb`** — apenas SEMP
- **`FROLS.ipynb`** — apenas FROLS
- **`NLSI.ipynb`** — apenas Gram-Schmidt

### Alterando o código Rust

1. Edite os arquivos em `rust_backend/src/`
2. Recompile: `cd rust_backend && python -m maturin build --release`
3. Reinstale o wheel: `pip install target/wheels/rust_nlsi-*.whl --force-reinstall`
4. Reinicie o kernel do notebook

## Uso via Python

```python
import pandas as pd
from src import Semp, Frols, GramSchmidt

df = pd.read_csv('data/exchanger.csv')
u, y = df['q'].values, df['th'].values

# SEMP
semp = Semp(u, y, l=1, nu=2, ny=2, ne=0)
semp.run(validation=True, title='Exchanger — SEMP')

# FROLS
frols = Frols(u, y, nu=5, ny=5, ne=0, nl=1, tol=0.0, max_iter=10)
frols.run(validation=True, title='Exchanger — FROLS')

# Gram-Schmidt
gs = GramSchmidt(u, y, nu=2, ny=2, nl=3, n_theta=4)
gs.run(validation=True, title='Exchanger — Gram-Schmidt')
```

---

# Descrição dos datasets

Os dados usados para testar os algoritmos foram obtidos online das bases [DaISy](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html) e [Nonlinear Benchmarks](https://www.nonlinearbenchmark.org/benchmarks).

## Ball and beam

Arquivo `ball-and-beam.csv` — practicum ball and beam do ESAT-SISTA.

- Amostragem: 0.1 s | Amostras: 1000
- Entrada `u`: ângulo do beam
- Saída `y`: posição da bola

## Liquid-saturated steam heat exchanger

Arquivo `exchanger.csv` — trocador de calor a vapor saturado.

- Amostragem: 1 s | Amostras: 4000
- Entrada `q`: vazão do líquido
- Saída `th`: temperatura de saída

## Flexible robot arm

Arquivo `robot-arm.csv` — braço robótico flexível.

- Amostras: 1024
- Entrada `u`: torque de reação
- Saída `y`: aceleração do braço flexível

## Cascaded tanks with overflow

Arquivo `tanque.csv` — tanques em cascata com overflow.

- Amostragem: 4 s | Amostras: 1024
- Entrada `uEst`: tensão
- Saída `yEst`: nível de água

## Silverbox system

Arquivo `SNLS80mV.csv` — oscilador de Duffing eletrônico.

- Amostras: 131073
- Entrada `V1`: sinal de entrada
- Saída `V2`: saída medida
