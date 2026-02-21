import langchain
import pydantic


class UnitContext(pydantic.BaseModel):
    pos_x: int = 0
    pos_y: int = 0
    has_sword: bool = False
    enemies: list[int] = pydantic.fields.Field(default_factory=lambda: [1, 2, 3])


@langchain.tools.tool
def make_one_step(
    runtime: langchain.tools.ToolRuntime[UnitContext], towards_x: int, towards_y: int
):
    """
    Do one step towards coordinates (towards_x, towards_y).
    This function changes current position by one cell in the direction towards destination.
    Always call this function in the loop multiple times until (towards_x, towards_y) is reached.
    """
    if delta := towards_x - runtime.context.pos_x:
        runtime.context.pos_x += abs(delta) // delta
    if delta := towards_y - runtime.context.pos_y:
        runtime.context.pos_y += abs(delta) // delta


@langchain.tools.tool
def get_current_position(
    runtime: langchain.tools.ToolRuntime[UnitContext],
) -> tuple[int, int]:
    """
    Returns current unit position as x,y coordinates.
    """
    return (runtime.context.pos_x, runtime.context.pos_y)


@langchain.tools.tool
def pick_sword(
    runtime: langchain.tools.ToolRuntime[UnitContext],
):
    """
    Unit picks a sword.
    """
    runtime.context.has_sword = True


@langchain.tools.tool
def drop_sword(
    runtime: langchain.tools.ToolRuntime[UnitContext],
):
    """
    Unit drops a sword.
    """
    runtime.context.has_sword = False


@langchain.tools.tool
def has_sword(
    runtime: langchain.tools.ToolRuntime[UnitContext],
) -> int:
    """
    Returns 1 if unit has a sword in the hands.
    """
    return int(runtime.context.has_sword)


@langchain.tools.tool
def get_enemies_around(
    runtime: langchain.tools.ToolRuntime[UnitContext],
) -> int:
    """
    Return closes enemy id around the unit. Returns 0 if no enemies found.
    """
    if runtime.context.enemies:
        return runtime.context.enemies.pop()
    return 0


@langchain.tools.tool
def attack_enemy(
    enemy_id: int,
    runtime: langchain.tools.ToolRuntime[UnitContext],
):
    """
    Attacks enemy with a specified id. Sword in hands needed to
    attack enemy.
    """
    pass
