# Responsabilidade: Define experimentos, janela de convergência e limiares da comparação.
#
# Contém configurações ou processamento no nível do módulo; não define funções próprias.

"""Strategy comparison — strategy_settings."""



EXPERIMENTS = [
    # --- OxfordPets ---
    ("OxfordPets", "LoRA Padrão",
     "fl_puma_clip_lora_prune1_ala1_paca12",
     "server_OxfordPets_*_run*.h5"),

    ("OxfordPets", "SoRA Estático",
     "fl_puma_OxfordPets_clip_sora_with_schedule_prune1_ala1_paca12",
     "server_OxfordPets_*_run*.h5"),

    ("OxfordPets", "PUMA-GT",
     "fl_puma_OxfordPets_clip_sora_with_schedule_prune1_ala1_adaptpaca",
     "server_OxfordPets_*_run*.h5"),

    # --- DTD ---
    ("DTD", "LoRA Padrão",
     "fl_puma_DTD_clip_lora_prune1_ala1_paca12",
     "server_DTD_*_run*.h5"),

    ("DTD", "SoRA Estático",
     "fl_puma_DTD_clip_sora_with_schedule_prune1_ala1_paca12",
     "server_DTD_*_run*.h5"),

    ("DTD", "PUMA-GT",
     "fl_puma_DTD_clip_sora_with_schedule_prune1_ala1_adaptpaca",
     "server_DTD_*_run*.h5"),

    # --- FGVCAircraft ---
    ("FGVCAircraft", "LoRA Padrão",
     "fl_puma_FGVCAircraft_clip_lora_prune1_ala1_paca12",
     "server_FGVCAircraft_*_run*.h5"),

    ("FGVCAircraft", "SoRA Estático",
     "fl_puma_FGVCAircraft_clip_sora_with_schedule_prune1_ala1_paca12",
     "server_FGVCAircraft_*_run*.h5"),

    ("FGVCAircraft", "PUMA-GT",
     "fl_puma_FGVCAircraft_clip_sora_with_schedule_prune1_ala1_adaptpaca",
     "server_FGVCAircraft_*_run*.h5"),

    # --- Flowers102 ---
    ("Flowers102", "LoRA Padrão",
     "fl_puma_Flowers102_clip_lora_prune1_ala1_paca12",
     "server_Flowers102_*_run*.h5"),

    ("Flowers102", "SoRA Estático",
     "fl_puma_Flowers102_clip_sora_with_schedule_prune1_ala1_paca12",
     "server_Flowers102_*_run*.h5"),

    ("Flowers102", "PUMA-GT",
     "fl_puma_Flowers102_clip_sora_with_schedule_prune1_ala1_adaptpaca",
     "server_Flowers102_*_run*.h5"),
]

CONVERGENCE_WINDOW = 20     # Tamanho da janela deslizante (em rodadas)

CONVERGENCE_THRESHOLD = 0.5 # Desvio padrão da acurácia dentro da janela (%)



