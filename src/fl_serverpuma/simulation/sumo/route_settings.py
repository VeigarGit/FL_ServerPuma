# Responsabilidade: Centraliza os padrões e caminhos de geração de rotas.
#
# Contém configurações ou processamento no nível do módulo; não define funções próprias.

"""SUMO route generation — route_settings."""



DEFAULT_SPEED = 13.89

INTERSECTION_PENALTY = 5.0

DEFAULT_COOLDOWN = 5

SAFETY_MARGIN = 1.3

DEFAULT_MIN_START_LENGTH = 15.0

MAX_BACKTRACK_ATTEMPTS = 50

MAX_ROUTE_RETRIES = 10

VEHICLE_COLORS = [
    "1,0,0",       # Vermelho
    "0,0.8,0",     # Verde
    "0,0,1",       # Azul
    "1,1,0",       # Amarelo
    "1,0,1",       # Magenta
    "0,1,1",       # Ciano
    "1,0.5,0",     # Laranja
    "0.5,0,1",     # Roxo
    "0,0.5,0",     # Verde escuro
    "1,0.5,0.5",   # Salmao
]



