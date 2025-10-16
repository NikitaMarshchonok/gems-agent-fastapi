import json, os, uuid
from typing import List
from .models import Gem

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "gems.json")

def _ensure_file():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    needs_seed = False
    if not os.path.exists(DATA_PATH):
        needs_seed = True
    else:
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            if not txt:
                needs_seed = True
            else:
                data = json.loads(txt)
                if not isinstance(data, list) or len(data) == 0:
                    needs_seed = True
        except Exception:
            needs_seed = True

    if needs_seed:
        seed = [
            Gem(
                id=str(uuid.uuid4()),
                name="Travel",
                system_prompt=(
                    "You are a world-class travel planner. Be concise, structured, and pragmatic. "
                    "When suggesting itineraries, include timings, logistics, and price hints. "
                    "Use tools if available."
                ),
                tools=["web_search", "calculator"],
                temperature=0.3
            ).model_dump(),
            Gem(
                id=str(uuid.uuid4()),
                name="Code Helper",
                system_prompt=(
                    "You are a senior Python developer. Explain step-by-step, show short code snippets, "
                    "and warn about edge cases. Keep answers focused."
                ),
                tools=["web_search"],
                temperature=0.2
            ).model_dump(),
            Gem(
                id=str(uuid.uuid4()),
                name="English Tutor",
                system_prompt=(
                    "You are a patient English tutor. Use simple language and provide two examples for each concept."
                ),
                tools=[],
                temperature=0.2
            ).model_dump()
        ]
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(seed, f, ensure_ascii=False, indent=2)

def load_all() -> List[Gem]:
    _ensure_file()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Gem(**x) for x in data]

def save_all(gems: List[Gem]) -> None:
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump([g.model_dump() for g in gems], f, ensure_ascii=False, indent=2)

def add_gem(gem: Gem) -> Gem:
    gems = load_all()
    gems.append(gem)
    save_all(gems)
    return gem

def update_gem(gem_id: str, patch: dict) -> Gem | None:
    gems = load_all()
    updated = None
    for i, g in enumerate(gems):
        if g.id == gem_id:
            data = g.model_dump()
            data.update({k: v for k, v in patch.items() if v is not None})
            updated = Gem(**data)
            gems[i] = updated
            break
    if updated:
        save_all(gems)
    return updated

def delete_gem(gem_id: str) -> bool:
    gems = load_all()
    new_gems = [g for g in gems if g.id != gem_id]
    if len(new_gems) == len(gems):
        return False
    save_all(new_gems)
    return True

def get_gem(gem_id: str) -> Gem | None:
    for g in load_all():
        if g.id == gem_id:
            return g
    return None

def new_id() -> str:
    return str(uuid.uuid4())
