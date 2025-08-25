from dataclasses import dataclass

from src.llm.session.operations.actor import ActorOperations
from src.llm.session.operations.injection import InjectionOperations
from src.llm.session.operations.round import RoundOperations
from src.llm.session.operations.turn import TurnOperations


@dataclass
class SessionOperations:
    turn: TurnOperations
    injection: InjectionOperations
    actor: ActorOperations
    round: RoundOperations
