import adsk.core
import adsk.fusion
import traceback
import math
import heapq
import time
import os

handlers = []

CMD_ID = "cmdPipeVoxelDebug"
CMD_NAME = "Voxel Pipe Path"
CMD_DESCRIPTION = "Generate pipe path using A* and voxelization. Select body, start point, and end point."
WORKSPACE_ID = "FusionSolidEnvironment"
PANEL_ID = "SolidCreatePanel"

GRID_CACHE = {}

command_definition = None
command_control = None


# default settings, these are overwritten at execution
VOXEL_SIZE = 0.4
WALL_CLEARANCE = 0.0
NUDGE_STEPS = 20
NUDGE_STEP_SCALE = 0.5


def log(msg: str):
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        palette = ui.palettes.itemById('TextCommands')
        if palette:
            palette.isVisible = True
            palette.writeText(str(msg))
    except:
        pass

# progress update helper, also checks for cancellation
def progress_update(progress, value, message=None):
    if progress:
        if message:
            progress.message = message
        progress.progressValue = value
        adsk.doEvents()

        if progress.wasCancelled:
            raise Exception("Operation cancelled by user.")

# helpers

# selects the body
def get_body(sel):
    ent = sel.entity
    if isinstance(ent, adsk.fusion.BRepBody):
        return ent
    if isinstance(ent, adsk.fusion.BRepFace):
        return ent.body
    raise ValueError("Select body")

# returns sketch point, used for start and end points
def get_sketch_point(sel):
    ent = sel.entity
    if isinstance(ent, adsk.fusion.SketchPoint):
        return ent
    raise ValueError("Select sketch point")

# checks if a point is inside the object
def point_inside(body, p):
    return body.pointContainment(p) == adsk.fusion.PointContainment.PointInsidePointContainment


def bbox_center(body):
    bb = body.boundingBox
    return adsk.core.Point3D.create(
        (bb.minPoint.x + bb.maxPoint.x) * 0.5,
        (bb.minPoint.y + bb.maxPoint.y) * 0.5,
        (bb.minPoint.z + bb.maxPoint.z) * 0.5
    )

# nudges points inside to avoid points on the surface
def nudge_inside(body, p):
    if point_inside(body, p):
        log(f"Point already inside: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
        return p

    center = bbox_center(body)
    v = p.vectorTo(center)

    if v.length < 1e-9:
        log("Nudge failed: point is at bbox center.")
        return p

    v.normalize()

    for i in range(1, NUDGE_STEPS + 1):
        test = p.copy()
        step = v.copy()
        step.scaleBy(VOXEL_SIZE * NUDGE_STEP_SCALE * i)
        test.translateBy(step)

        if point_inside(body, test):
            log(
                f"Nudged point inside after {i} steps: "
                f"({test.x:.3f}, {test.y:.3f}, {test.z:.3f})"
            )
            return test

    log("WARNING: Could not nudge point inside.")
    return p


# grid

# cache
def get_body_cache_key(body):
    bb = body.boundingBox

    return (
        body.entityToken,
        round(bb.minPoint.x, 6),
        round(bb.minPoint.y, 6),
        round(bb.minPoint.z, 6),
        round(bb.maxPoint.x, 6),
        round(bb.maxPoint.y, 6),
        round(bb.maxPoint.z, 6),
        round(VOXEL_SIZE, 6),
        round(WALL_CLEARANCE, 6)
    )

class Grid:
    def __init__(self, origin, nx, ny, nz):
        self.origin = origin
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.valid = [False] * (nx * ny * nz)

    # returns a flat self
    def flatten(self, idx):
        i, j, k = idx
        return k * self.nx * self.ny + j * self.nx + i

    # checks if a point is contained
    def in_bounds(self, idx):
        i, j, k = idx
        return 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz

    def center(self, idx):
        i, j, k = idx
        return adsk.core.Point3D.create(
            self.origin.x + (i + 0.5) * VOXEL_SIZE,
            self.origin.y + (j + 0.5) * VOXEL_SIZE,
            self.origin.z + (k + 0.5) * VOXEL_SIZE
        )

    def point_to_index(self, p):
        i = int((p.x - self.origin.x) / VOXEL_SIZE)
        j = int((p.y - self.origin.y) / VOXEL_SIZE)
        k = int((p.z - self.origin.z) / VOXEL_SIZE)
        return (i, j, k)

    def is_valid(self, idx):
        return self.valid[self.flatten(idx)]


# voxelization
def build_grid(body, progress=None):
    bb = body.boundingBox

    origin = adsk.core.Point3D.create(
        bb.minPoint.x - VOXEL_SIZE,
        bb.minPoint.y - VOXEL_SIZE,
        bb.minPoint.z - VOXEL_SIZE
    )

    # body.boundingBox has an xyz max + min which we use to make the starting grid
    nx = int((bb.maxPoint.x - bb.minPoint.x) / VOXEL_SIZE) + 3
    ny = int((bb.maxPoint.y - bb.minPoint.y) / VOXEL_SIZE) + 3
    nz = int((bb.maxPoint.z - bb.minPoint.z) / VOXEL_SIZE) + 3

    grid = Grid(origin, nx, ny, nz)

    log(
        f"Building grid: nx={nx}, ny={ny}, nz={nz}, "
        f"voxel={VOXEL_SIZE}, clearance={WALL_CLEARANCE}"
    )

    # pass 1: mark all inside voxels
    inside = [False] * (nx * ny * nz)
    inside_count = 0

    for k in range(nz):
        progress_update(
            progress,
            int((k / max(nz - 1, 1)) * 40),
            f"Building grid... layer {k + 1} of {nz}"
        )

        for j in range(ny):
            for i in range(nx):
                idx = (i, j, k)
                p = grid.center(idx)

                if point_inside(body, p):
                    inside[grid.flatten(idx)] = True
                    inside_count += 1

    log(f"Initial inside voxels: {inside_count} / {nx * ny * nz}")

    # no clearance requested
    if WALL_CLEARANCE <= 1e-9:
        grid.valid = inside
        log("No wall clearance applied.")
        progress_update(progress, 70, "No wall clearance applied.")
        return grid

    # pass 2: erode by clearance distance
    # This converts the physical clearance distance into a voxel-neighborhood radius.
    # if Wall Clearance is 4mm and Voxel Size is 1mm, then clearance_vox will be 4, 
    # meaning we check a 9x9x9 cube of voxels around each voxel.
    clearance_vox = int(math.ceil(WALL_CLEARANCE / VOXEL_SIZE))
    kept_count = 0

    offsets = []
    # If any nearby voxel center within that radius is outside the body, 
    # then this voxel is too close to a wall, hole, or boundary, and it is rejected.
    for dk in range(-clearance_vox, clearance_vox + 1):
        for dj in range(-clearance_vox, clearance_vox + 1):
            for di in range(-clearance_vox, clearance_vox + 1):
                dist = math.sqrt(
                    (di * VOXEL_SIZE) ** 2 +
                    (dj * VOXEL_SIZE) ** 2 +
                    (dk * VOXEL_SIZE) ** 2
                )
                if dist <= WALL_CLEARANCE:
                    offsets.append((di, dj, dk))

    log(f"Applying clearance erosion with {len(offsets)} neighbor offsets")

    for k in range(nz):
        progress_update(
            progress,
            40 + int((k / max(nz - 1, 1)) * 30),
            f"Checking wall clearance... layer {k + 1} of {nz}"
        )

        for j in range(ny):
            for i in range(nx):
                idx = (i, j, k)
                flat = grid.flatten(idx)

                if not inside[flat]:
                    continue

                safe = True

                for di, dj, dk in offsets:
                    ni = i + di
                    nj = j + dj
                    nk = k + dk
                    nidx = (ni, nj, nk)

                    # If any nearby voxel within clearance radius is outside,
                    # then this voxel is too close to the wall/hole boundary.
                    if not grid.in_bounds(nidx):
                        safe = False
                        break

                    if not inside[grid.flatten(nidx)]:
                        safe = False
                        break

                if safe:
                    grid.valid[flat] = True
                    kept_count += 1

    log(f"Clearance-applied valid voxels: {kept_count} / {nx * ny * nz}")
    progress_update(progress, 70, "Wall clearance check complete.")
    return grid


# endpoint mapping

def nearest_valid_voxel(grid, seed, max_radius=8):
    if seed is None:
        return None

    if not grid.in_bounds(seed):
        log(f"Seed out of bounds: {seed}")
        return None

    if grid.is_valid(seed):
        log(f"Seed is already valid: {seed}")
        return seed

    log(f"Seed not valid, searching nearest valid voxel from {seed}")

    si, sj, sk = seed

    for r in range(1, max_radius + 1):
        for k in range(sk - r, sk + r + 1):
            for j in range(sj - r, sj + r + 1):
                for i in range(si - r, si + r + 1):
                    idx = (i, j, k)
                    if not grid.in_bounds(idx):
                        continue
                    if grid.is_valid(idx):
                        log(f"Nearest valid voxel found at radius {r}: {idx}")
                        return idx

    log("No nearby valid voxel found.")
    return None


# A*
# distance from node to end goal as euclidian distance 
def heuristic(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def neighbors(idx):
    i, j, k = idx
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue
                yield (i + di, j + dj, k + dk)


def astar(grid, start, goal, progress=None):
    log(f"A* start={start}, goal={goal}")

    if start is None or goal is None:
        log("A* stopped: start or goal is None")
        return None

    if not grid.in_bounds(start):
        log("A* stopped: start out of bounds")
        return None
    if not grid.in_bounds(goal):
        log("A* stopped: goal out of bounds")
        return None
    if not grid.is_valid(start):
        log("A* stopped: start voxel is not valid")
        return None
    if not grid.is_valid(goal):
        log("A* stopped: goal voxel is not valid")
        return None

    progress_update(progress, 70, "Searching with A*...")

    open_heap = []
    heapq.heappush(open_heap, (0.0, start))

    came = {}
    g = {start: 0.0}
    explored = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        explored += 1

        if explored % 500 == 0:
            log(f"A* explored {explored} nodes, current={current}")

            progress_value = 70 + min(29, explored // 500)
            progress_update(
                progress,
                progress_value,
                f"Searching with A*... explored {explored} nodes"
            )

        if current == goal:
            log(f"A* reached goal after exploring {explored} nodes")
            break

        for nxt in neighbors(current):
            if not grid.in_bounds(nxt):
                continue
            if not grid.is_valid(nxt):
                continue

            step_cost = math.sqrt(
                (nxt[0] - current[0]) ** 2 +
                (nxt[1] - current[1]) ** 2 +
                (nxt[2] - current[2]) ** 2
            )

            tentative = g[current] + step_cost

            if nxt not in g or tentative < g[nxt]:
                came[nxt] = current
                g[nxt] = tentative
                f = tentative + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, nxt))

    if goal not in came and goal != start:
        log(f"A* failed. Explored {explored} nodes and did not reach goal.")
        return None

    path = [goal]
    while path[-1] != start:
        path.append(came[path[-1]])
    path.reverse()

    log(f"A* path length: {len(path)}")
    progress_update(progress, 100, "Path found.")
    return path

# path smoothing by removing unnecessary intermediate points in straight segments
def simplify_path(path):
    if not path or len(path) <= 2:
        return path

    simplified = [path[0]]

    prev_dir = None

    for i in range(1, len(path)):
        a = path[i - 1]
        b = path[i]

        direction = (
            b[0] - a[0],
            b[1] - a[1],
            b[2] - a[2]
        )

        if prev_dir is None:
            prev_dir = direction
            continue

        if direction != prev_dir:
            simplified.append(path[i - 1])
            prev_dir = direction

    simplified.append(path[-1])
    return simplified


# command handling

class CommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            cmd = args.command
            inputs = cmd.commandInputs

            body = inputs.addSelectionInput("body", "Body", "Select body")
            body.addSelectionFilter("Bodies")
            body.setSelectionLimits(1, 1)

            p1 = inputs.addSelectionInput("p1", "Start Point", "Select sketch point")
            p1.addSelectionFilter("SketchPoints")
            p1.setSelectionLimits(1, 1)

            p2 = inputs.addSelectionInput("p2", "End Point", "Select sketch point")
            p2.addSelectionFilter("SketchPoints")
            p2.setSelectionLimits(1, 1)

            inputs.addValueInput(
                "voxel",
                "Voxel Size",
                "mm",
                adsk.core.ValueInput.createByString("4 mm")
            )

            inputs.addValueInput(
                "clearance",
                "Minimum Wall Distance",
                "mm",
                adsk.core.ValueInput.createByString("4 mm")
            )

            midpoint = inputs.addSelectionInput("midpoint", "Midpoint", "Optional sketch point the path must go through")
            midpoint.addSelectionFilter("SketchPoints")
            midpoint.setSelectionLimits(0, 1)

            inputs.addBoolValueInput(
                "smooth",
                "Smooth Path",
                True,
                "",
                True
            )

            inputs.addBoolValueInput(
                "use_cache",
                "Use Cache",
                True,
                "",
                False
            )

            on_exec = CommandExecuteHandler()
            cmd.execute.add(on_exec)
            handlers.append(on_exec)

            on_select = CommandSelectHandler()
            cmd.select.add(on_select)
            handlers.append(on_select)

            body.hasFocus = True

        except:
            adsk.core.Application.get().userInterface.messageBox(traceback.format_exc())

class CommandSelectHandler(adsk.core.SelectionEventHandler):
    def notify(self, args):
        try:
            cmd = args.firingEvent.sender
            inputs = cmd.commandInputs

            body_input = adsk.core.SelectionCommandInput.cast(inputs.itemById("body"))
            p1_input = adsk.core.SelectionCommandInput.cast(inputs.itemById("p1"))
            p2_input = adsk.core.SelectionCommandInput.cast(inputs.itemById("p2"))
            voxel_input = adsk.core.ValueCommandInput.cast(inputs.itemById("voxel"))

            if body_input.selectionCount > 0 and p1_input.selectionCount == 0:
                adsk.doEvents()
                p1_input.hasFocus = True

            elif p1_input.selectionCount > 0 and p2_input.selectionCount == 0:
                adsk.doEvents()
                p2_input.hasFocus = True

            elif p2_input.selectionCount > 0:
                adsk.doEvents()
                voxel_input.hasFocus = True
        except:
            adsk.core.Application.get().userInterface.messageBox(traceback.format_exc())

class CommandExecuteHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        app = adsk.core.Application.get()
        ui = app.userInterface
        timer_start = time.perf_counter()
        progress = None

        try:
            global VOXEL_SIZE, WALL_CLEARANCE

            cmd = args.firingEvent.sender
            inputs = cmd.commandInputs

            body_sel = adsk.core.SelectionCommandInput.cast(inputs.itemById("body")).selection(0)
            p1_sel = adsk.core.SelectionCommandInput.cast(inputs.itemById("p1")).selection(0)
            p2_sel = adsk.core.SelectionCommandInput.cast(inputs.itemById("p2")).selection(0)
            midpoint_input = adsk.core.SelectionCommandInput.cast(inputs.itemById("midpoint"))
            midpoint_sel = midpoint_input.selection(0) if midpoint_input.selectionCount > 0 else None

            body = get_body(body_sel)
            sp1 = get_sketch_point(p1_sel)
            sp2 = get_sketch_point(p2_sel)
            sp_mid = get_sketch_point(midpoint_sel) if midpoint_sel else None

            VOXEL_SIZE = adsk.core.ValueCommandInput.cast(inputs.itemById("voxel")).value
            WALL_CLEARANCE = adsk.core.ValueCommandInput.cast(inputs.itemById("clearance")).value
            smooth_enabled = adsk.core.BoolValueCommandInput.cast(inputs.itemById("smooth")).value
            cache_enabled = adsk.core.BoolValueCommandInput.cast(inputs.itemById("use_cache")).value

            progress = ui.createProgressDialog()
            progress.cancelButtonText = "Cancel"
            progress.isBackgroundTranslucent = False
            progress.show(
                "Voxel Pipe Path",
                "Starting...",
                0,
                100,
                1
            )

            p1_raw = sp1.worldGeometry
            p2_raw = sp2.worldGeometry
            midpoint_raw = sp_mid.worldGeometry if sp_mid else None

            log("=== Pipe Path Debug Start ===")
            log(f"Voxel size = {VOXEL_SIZE}")
            log(f"Wall clearance = {WALL_CLEARANCE}")
            log(f"Smooth path = {smooth_enabled}")
            log(f"Use cache = {cache_enabled}")
            log(f"Raw p1 = ({p1_raw.x:.3f}, {p1_raw.y:.3f}, {p1_raw.z:.3f})")
            log(f"Raw p2 = ({p2_raw.x:.3f}, {p2_raw.y:.3f}, {p2_raw.z:.3f})")
            if midpoint_raw:
                log(f"Raw midpoint = ({midpoint_raw.x:.3f}, {midpoint_raw.y:.3f}, {midpoint_raw.z:.3f})")
            log(f"p1 inside? {point_inside(body, p1_raw)}")
            log(f"p2 inside? {point_inside(body, p2_raw)}")
            if midpoint_raw:
                log(f"midpoint inside? {point_inside(body, midpoint_raw)}")

            p1 = nudge_inside(body, p1_raw)
            p2 = nudge_inside(body, p2_raw)
            midpoint = nudge_inside(body, midpoint_raw) if midpoint_raw else None

            log(f"Nudged p1 = ({p1.x:.3f}, {p1.y:.3f}, {p1.z:.3f})")
            log(f"Nudged p2 = ({p2.x:.3f}, {p2.y:.3f}, {p2.z:.3f})")
            if midpoint:
                log(f"Nudged midpoint = ({midpoint.x:.3f}, {midpoint.y:.3f}, {midpoint.z:.3f})")
            log(f"Nudged p1 inside? {point_inside(body, p1)}")
            log(f"Nudged p2 inside? {point_inside(body, p2)}")
            if midpoint:
                log(f"Nudged midpoint inside? {point_inside(body, midpoint)}")

            cache_key = get_body_cache_key(body)

            if cache_enabled and cache_key in GRID_CACHE:
                log("Using cached grid.")
                progress_update(progress, 70, "Using cached grid.")
                grid = GRID_CACHE[cache_key]
            else:
                if cache_enabled:
                    log("No cached grid found. Building new grid.")
                else:
                    log("Cache disabled. Building new grid.")

                grid = build_grid(body, progress)

                if cache_enabled:
                    GRID_CACHE[cache_key] = grid

                    while len(GRID_CACHE) > 5:
                        oldest_key = next(iter(GRID_CACHE))
                        del GRID_CACHE[oldest_key]

                    log("Grid cached.")

            start_seed = grid.point_to_index(p1)
            goal_seed = grid.point_to_index(p2)
            midpoint_seed = grid.point_to_index(midpoint) if midpoint else None

            log(f"Start seed index = {start_seed}, in bounds? {grid.in_bounds(start_seed)}")
            log(f"Goal seed index = {goal_seed}, in bounds? {grid.in_bounds(goal_seed)}")
            if midpoint_seed:
                log(f"Midpoint seed index = {midpoint_seed}, in bounds? {grid.in_bounds(midpoint_seed)}")

            if grid.in_bounds(start_seed):
                log(f"Start seed valid? {grid.is_valid(start_seed)}")
            if grid.in_bounds(goal_seed):
                log(f"Goal seed valid? {grid.is_valid(goal_seed)}")
            if midpoint_seed and grid.in_bounds(midpoint_seed):
                log(f"Midpoint seed valid? {grid.is_valid(midpoint_seed)}")

            start = nearest_valid_voxel(grid, start_seed)
            goal = nearest_valid_voxel(grid, goal_seed)
            midpoint_voxel = nearest_valid_voxel(grid, midpoint_seed) if midpoint_seed else None

            log(f"Mapped start voxel = {start}")
            log(f"Mapped goal voxel = {goal}")
            if midpoint_seed:
                log(f"Mapped midpoint voxel = {midpoint_voxel}")

            if start is None or goal is None or (midpoint_seed and midpoint_voxel is None):
                ui.messageBox(
                    "Could not map start, midpoint, or goal to a valid voxel.\n"
                    "Try reducing Minimum Wall Distance or Voxel Size.\n"
                    "Open Text Commands for details."
                )
                return

            if midpoint_voxel:
                log("Midpoint set. Running A* from start to midpoint, then midpoint to goal.")
                path_to_midpoint = astar(grid, start, midpoint_voxel, progress)

                if not path_to_midpoint:
                    ui.messageBox(
                        "No path found from start point to midpoint.\n"
                        "Try reducing Minimum Wall Distance or Voxel Size.\n"
                        "Open Text Commands for debug output."
                    )
                    return

                path_from_midpoint = astar(grid, midpoint_voxel, goal, progress)

                if not path_from_midpoint:
                    ui.messageBox(
                        "No path found from midpoint to end point.\n"
                        "Try reducing Minimum Wall Distance or Voxel Size.\n"
                        "Open Text Commands for debug output."
                    )
                    return

                path = path_to_midpoint + path_from_midpoint[1:]
            else:
                path = astar(grid, start, goal, progress)

                if not path:
                    ui.messageBox(
                        "No path found.\n"
                        "Try reducing Minimum Wall Distance or Voxel Size.\n"
                        "Open Text Commands for debug output."
                    )
                    return

            if smooth_enabled:
                if midpoint_voxel:
                    final_path_to_midpoint = simplify_path(path_to_midpoint)
                    final_path_from_midpoint = simplify_path(path_from_midpoint)
                    final_path = final_path_to_midpoint + final_path_from_midpoint[1:]
                    log(f"Simplified path length: {len(final_path)} from {len(path)} with midpoint preserved")
                else:
                    final_path = simplify_path(path)
                    log(f"Simplified path length: {len(final_path)} from {len(path)}")
            else:
                final_path = path
                log("Path smoothing disabled.")

            design = adsk.fusion.Design.cast(app.activeProduct)
            root = design.rootComponent

            sk = root.sketches.add(root.xYConstructionPlane)
            sk.is3D = True

            pts = adsk.core.ObjectCollection.create()

            # start at the actual selected start point
            pts.add(p1_raw)

            # then go through the voxel-grid path
            for idx in final_path:
                pts.add(grid.center(idx))

            # end at the actual selected end point
            pts.add(p2_raw)

            sk.sketchCurves.sketchFittedSplines.add(pts)

            ui.messageBox(
                f"Path created.\n"
                f"Path voxels: {len(path)}\n"
                f"Path points used: {len(final_path)}\n"
                f"Midpoint used: {midpoint_voxel is not None}\n"
                f"Smooth path: {smooth_enabled}\n"
                f"Open Text Commands for debug output."
            )

        except Exception as e:
            if str(e) == "Operation cancelled by user.":
                ui.messageBox("Operation cancelled.")
            else:
                ui.messageBox("Execute failed:\n" + traceback.format_exc())
        finally:
            if progress:
                progress.hide()

            elapsed = time.perf_counter() - timer_start
            log(f"Total execution time: {elapsed:.3f} seconds")


def run(context):
    global command_definition, command_control

    app = adsk.core.Application.get()
    ui = app.userInterface

    try:
        old = ui.commandDefinitions.itemById(CMD_ID)
        if old:
            old.deleteMe()

        icon_folder = os.path.join(os.path.dirname(__file__), "resources")

        command_definition = ui.commandDefinitions.addButtonDefinition(
            CMD_ID,
            CMD_NAME,
            CMD_DESCRIPTION,
            icon_folder
        )

        on_created = CommandCreatedHandler()
        command_definition.commandCreated.add(on_created)
        handlers.append(on_created)

        workspace = ui.workspaces.itemById(WORKSPACE_ID)
        panel = workspace.toolbarPanels.itemById(PANEL_ID)

        old_control = panel.controls.itemById(CMD_ID)
        if old_control:
            old_control.deleteMe()

        command_control = panel.controls.addCommand(command_definition)
        command_control.isPromoted = True

    except:
        ui.messageBox("Run failed:\n" + traceback.format_exc())


def stop(context):
    global command_definition, command_control

    app = adsk.core.Application.get()
    ui = app.userInterface

    try:
        workspace = ui.workspaces.itemById(WORKSPACE_ID)
        panel = workspace.toolbarPanels.itemById(PANEL_ID)

        if panel:
            old_control = panel.controls.itemById(CMD_ID)
            if old_control:
                old_control.deleteMe()

        old = ui.commandDefinitions.itemById(CMD_ID)
        if old:
            old.deleteMe()

        command_control = None
        command_definition = None
        handlers.clear()

    except:
        ui.messageBox("Stop failed:\n" + traceback.format_exc())