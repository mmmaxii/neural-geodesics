# Neural Geodesics — Arquitectura del Sistema

## 1. Vista General del Pipeline

```mermaid
graph LR
    subgraph FASE1["Fase 1: Física"]
        A["Métrica de<br/>Schwarzschild"] --> B["Ecuación de Binet<br/>d²u/dφ² + u = 3Mu²"]
        B --> C["Integrador RK45"]
        C --> D["Dataset<br/>100K geodésicas"]
    end

    subgraph FASE2["Fase 2: Entrenamiento"]
        D --> E["GeodesicDataset<br/>normalización + sampling"]
        E --> F["PIMLP"]
        G["Physics Loss<br/>ODE residual"] --> F
        F --> H["Modelo Entrenado<br/>.pt"]
    end

    subgraph FASE3["Fase 3: Comparación"]
        H --> I["Neural Renderer"]
        C --> J["Classical Renderer"]
        I --> K["Comparación<br/>lado a lado"]
        J --> K
    end

    subgraph FASE4["Fase 4: Web"]
        H --> L["Export ONNX<br/>+ INT8"]
        L --> M["Browser<br/>WebGPU"]
    end

    style FASE1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style FASE2 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style FASE3 fill:#1a1a2e,stroke:#533483,color:#fff
    style FASE4 fill:#1a1a2e,stroke:#16c79a,color:#fff
```

---

## 2. Generación de Datos (Fase 1 — detalle)

```mermaid
graph TD
    subgraph INPUT["Entrada"]
        B_VALUES["Valores de b<br/>b ∈ [0.1 rₛ, 15 rₛ]"]
    end

    subgraph SAMPLING["Estrategia de Muestreo"]
        S1["Uniforme<br/>40K puntos"]
        S2["Log-espaciado<br/>30K puntos"]
        S3["Concentrado en b_crit<br/>30K puntos<br/>Zona crítica"]
    end

    subgraph ODE["Integración de la ODE"]
        BINET["Sistema de Binet<br/>du/dφ = p<br/>dp/dφ = −u + 3Mu²"]
        RK45["scipy.solve_ivp<br/>método: RK45"]
        EVENTS["Detección de Eventos"]
    end

    subgraph EVENTS_DETAIL["Eventos"]
        EV1["Captura<br/>u > 1/rₛ"]
        EV2["Escape<br/>r > r_max"]
        EV3["Disco<br/>intersección"]
    end

    subgraph OUTPUT["Dataset de Salida (.npz)"]
        OUT1["b_values<br/>float32"]
        OUT2["deflection_angles<br/>Δφ float32"]
        OUT3["captured_flags<br/>bool"]
        OUT4["trajectories<br/>lista de arrays"]
    end

    B_VALUES --> S1 & S2 & S3
    S1 & S2 & S3 --> BINET
    BINET --> RK45
    RK45 --> EVENTS
    EVENTS --> EV1 & EV2 & EV3
    EV1 & EV2 & EV3 --> OUT1 & OUT2 & OUT3 & OUT4

    style S3 fill:#e94560,stroke:#e94560,color:#fff
    style EV1 fill:#c0392b,stroke:#c0392b,color:#fff
    style EV2 fill:#27ae60,stroke:#27ae60,color:#fff
    style EV3 fill:#f39c12,stroke:#f39c12,color:#fff
```

---

## 3. Arquitectura PIMLP (Fase 2 — detalle)

```mermaid
graph LR
    subgraph INPUT_LAYER["Entrada"]
        IN["b / rₛ<br/>(1 neurona)"]
    end

    subgraph HIDDEN["Capas Ocultas + Residual"]
        H1["Dense 64<br/>+ SiLU"]
        H2["Dense 128<br/>+ SiLU"]
        H3["Dense 128<br/>+ SiLU"]
        H4["Dense 64<br/>+ SiLU"]
    end

    subgraph OUTPUT_HEADS["Salidas"]
        HEAD1["Dense 1<br/>(sin activación)<br/>──────────<br/>Δφ deflexión"]
        HEAD2["Dense 1<br/>+ Sigmoid<br/>──────────<br/>P(captura)"]
    end

    IN --> H1 --> H2 --> H3 --> H4
    H2 -.->|"conexión<br/>residual"| H4
    H4 --> HEAD1
    H4 --> HEAD2

    style IN fill:#3498db,stroke:#3498db,color:#fff
    style HEAD1 fill:#e74c3c,stroke:#e74c3c,color:#fff
    style HEAD2 fill:#f39c12,stroke:#f39c12,color:#fff
    style H2 fill:#2c3e50,stroke:#8e44ad,color:#fff
    style H4 fill:#2c3e50,stroke:#8e44ad,color:#fff
```

**Total de parámetros**: ~35K (modelo muy ligero -> rápido en browser)

---

## 4. Función de Pérdida Híbrida

```mermaid
graph TD
    subgraph LOSS_TOTAL["𝓛 total = λ_d · 𝓛_data + λ_p · 𝓛_physics + λ_c · 𝓛_class"]

        subgraph DATA_LOSS["𝓛 data (MSE)"]
            DL["|| Δφ_pred − Δφ_RK45 ||²"]
        end

        subgraph PHYSICS_LOSS["𝓛 physics (ODE Residual)"]
            PL["|| d²û/dφ² + û − 3Mû² ||²<br/>────────────────────<br/>evaluado con torch.autograd<br/>en puntos de colocación"]
        end

        subgraph CLASS_LOSS["𝓛 class (BCE)"]
            CL["BCE(P_pred, capturado_real)<br/>────────────────────<br/>Binary Cross-Entropy"]
        end
    end

    subgraph SCHEDULING["Lambda Scheduling"]
        SCH["Época 0→50: λ_p bajo (aprende datos)<br/>Época 50→150: λ_p sube (aprende física)<br/>Época 150→200: λ_p alto (refina)"]
    end

    DATA_LOSS & PHYSICS_LOSS & CLASS_LOSS --> SCHEDULING

    style DATA_LOSS fill:#2980b9,stroke:#2980b9,color:#fff
    style PHYSICS_LOSS fill:#c0392b,stroke:#c0392b,color:#fff
    style CLASS_LOSS fill:#f39c12,stroke:#f39c12,color:#fff
```

> **¿Por qué "Physics-Informed"?** La 𝓛_physics fuerza a la red a respetar la ecuación de Binet. Sin esto, la red podría dar predicciones rápidas pero físicamente incorrectas. Con esto, incluso en zonas con pocos datos de entrenamiento, la red "sabe" la física.

---

## 5. Pipeline de Inferencia — Clásico vs Neural

```mermaid
graph TD
    PIXEL["Pixel (i, j)"] --> IMPACT["Calcular b<br/>del pixel"]

    IMPACT --> CLASSICAL_PATH
    IMPACT --> NEURAL_PATH

    subgraph CLASSICAL_PATH["Método Clásico (~ms por pixel)"]
        C1["Integrar ODE<br/>~100-1000 pasos RK45"]
        C2["¿Capturado?"]
        C3["Calcular Δφ"]
        C4["Buscar color<br/>en fondo/disco"]
        C1 --> C2 --> C3 --> C4
    end

    subgraph NEURAL_PATH["Método Neural (~μs por pixel)"]
        N1["Forward pass<br/>PIMLP(b/rₛ)"]
        N2["Δφ, P(captura)"]
        N3["Buscar color<br/>en fondo/disco"]
        N1 --> N2 --> N3
    end

    C4 --> RGB1["RGB clásico"]
    N3 --> RGB2["RGB neural"]

    RGB1 --> COMPARE["Comparación<br/>PSNR, SSIM, MAE<br/>Speedup: ~100×"]
    RGB2 --> COMPARE

    style CLASSICAL_PATH fill:#1a1a2e,stroke:#e74c3c,color:#fff
    style NEURAL_PATH fill:#1a1a2e,stroke:#2ecc71,color:#fff
    style COMPARE fill:#8e44ad,stroke:#8e44ad,color:#fff
```

---

## 6. Despliegue Web (Fase 4)

```mermaid
graph TD
    subgraph BACKEND["Backend (PyTorch)"]
        MODEL["Modelo PIMLP<br/>entrenado (.pt)"]
        EXPORT["torch → ONNX"]
        QUANT["Cuantización INT8"]
        MODEL --> EXPORT --> QUANT
    end

    subgraph BROWSER["Browser del Usuario"]
        ONNX_WEB["ONNX Runtime Web<br/>+ WebGPU"]
        SHADER["GLSL Shader<br/>(render clásico GPU)"]
        R3F["React Three Fiber<br/>+ Three.js"]
        CONTROLS["Controles (leva)<br/>• Masa<br/>• Cámara<br/>• Disco ON/OFF<br/>• FOV"]

        ONNX_WEB --> R3F
        SHADER --> R3F
        CONTROLS --> R3F
    end

    subgraph DISPLAY["Pantalla"]
        VIEW1["Render Neural<br/>~60 FPS"]
        VIEW2["Render Clásico<br/>~5 FPS"]
        METRICS["Métricas<br/>Error, FPS, Speedup"]
        VIEW1 & VIEW2 --> METRICS
    end

    QUANT -->|".onnx file<br/>(~150 KB)"| ONNX_WEB
    R3F --> VIEW1 & VIEW2

    style BACKEND fill:#1a1a2e,stroke:#3498db,color:#fff
    style BROWSER fill:#1a1a2e,stroke:#2ecc71,color:#fff
    style DISPLAY fill:#1a1a2e,stroke:#e74c3c,color:#fff
```

---

## 7. Estructura de Módulos

```mermaid
graph TD
    subgraph PROJECT["neural-geodesics"]
        subgraph PHYSICS["src/physics/"]
            P1["schwarzschild.py<br/>Métrica gμν"]
            P2["integrator.py<br/>ODE solver"]
            P3["ray_tracer.py<br/>Render clásico"]
            P4["accretion_disk.py<br/>Disco"]
            P1 --> P2 --> P3
            P1 --> P4
            P4 --> P3
        end

        subgraph NEURAL["src/neural/"]
            N1["model.py<br/>PIMLP"]
            N2["dataset.py<br/>Datos"]
            N3["loss.py<br/>𝓛 híbrida"]
            N4["train.py<br/>Trainer"]
            N5["export.py<br/>ONNX"]
            N2 --> N4
            N1 --> N4
            N3 --> N4
            N1 --> N5
        end

        subgraph RENDER["src/rendering/"]
            R1["classical_renderer.py"]
            R2["neural_renderer.py"]
            R3["comparison.py"]
            R1 --> R3
            R2 --> R3
        end

        P2 -.->|"genera datos"| N2
        P3 -.->|"render base"| R1
        N1 -.->|"modelo"| R2
    end

    style PHYSICS fill:#1a1a2e,stroke:#e94560,color:#fff
    style NEURAL fill:#1a1a2e,stroke:#3498db,color:#fff
    style RENDER fill:#1a1a2e,stroke:#2ecc71,color:#fff
```
