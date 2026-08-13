# gui.py
"""
Premium Light-Theme GUI — Tourist Route Planner (Project 26)
All 6 Modules visualised:
  1. Environment & State Space  (grid + landmarks)
  2. Search Algorithms           (BFS / DFS / UCS / A*)
  3. CSP Engine                  (budget + time sliders)
  4. Decision Making             (minimax stats)
  5. Uncertainty                 (Bayesian live probabilities)
  6. Integrated Pipeline         (XAI trace panel)
"""

import tkinter as tk
from tkinter import ttk
import time, random

from environment  import SpatialEnvironment
from constraints  import StructuralConstraintEngine
from uncertainty  import BayesianUncertaintyEngine
from decision     import SequentialDecisionMatrix, ExplainablePipelineEngine
import search

# ─── Colour Palette ──────────────────────────────────────────────────────────
C = {
    "bg":          "#0D0D0D",    # Level 0 Base - Obsidian absolute black
    "card":        "#1A1A1A",    # Level 1 Surfaces - Deep charcoal/obsidian
    "border":      "#2A2A2A",    # 1px solid border
    "header":      "#1F1F3D",    # Deep indigo header background
    "header_txt":  "#E5E2E1",    # Neon/light text
    
    # Grid cells
    "free":        "#131313",    # Idle cells dark
    "start":       "#6366f1",    # Electric Indigo
    "goal":        "#f59e0b",    # Amber
    "obstacle":    "#2A2A2A",    # Gray/black
    "traffic":     "#eab308",    # Amber/Yellow traffic
    "explored":    "#1F2937",    # Darker blue explored path
    "path":        "#22d3ee",    # Cyan active path
    "robot":       "#6366f1",    # Robot Indigo pulse
    "landmark":    "#ec4899",    # Magenta landmark highlight
    
    # Module accent strips (Obsidian Neon style)
    "m1": "#6366f1",   # Electric Indigo
    "m2": "#22d3ee",   # Cyan
    "m3": "#f59e0b",   # Amber
    "m4": "#10b981",   # Emerald Success
    "m5": "#8b5cf6",   # Purple
    "m6": "#ec4899",   # Magenta
    
    "txt":    "#E5E2E1",    # Light gray text
    "txt2":   "#A0A0B0",    # Muted secondary text
    "white":  "#FFFFFF",
    "primary":   "#6366f1", # Electric Indigo
    "secondary": "#22d3ee", # Cyan
    "accent":    "#ec4899", # Magenta
}

CELL_PX = 22
ANIM_EXPLORE = 6
ANIM_PATH    = 28
ANIM_ROBOT   = 40

# Fixed landmark positions (never change with Start/Goal)
LANDMARKS = {
    (0, 0):   ("Beach",   "🏖"),
    (3, 5):   ("Museum",  "🏛"),
    (10, 12): ("Temple",  "⛩"),
    (14, 8):  ("Park",    "🌳"),
    (19, 19): ("Mall",    "🛍"),
}


# Interaction modes
MODES = {
    "SET_START":  ("🟢 Set Start",  C["start"]),
    "SET_GOAL":   ("🔴 Set Goal",   C["goal"]),
    "OBSTACLE":   ("⬛ Obstacle",   C["obstacle"]),
    "TRAFFIC":    ("🟡 Traffic",    C["traffic"]),
    "ERASE":      ("✕  Erase",      C["m6"]),
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def pill_btn(parent, text, color, cmd, w=150, h=32):
    """Rounded pill button using Canvas."""
    cv = tk.Canvas(parent, width=w, height=h, bg=C["card"],
                   highlightthickness=0, cursor="hand2")
    r = h // 2

    def darken(hx, f=0.82):
        hx = hx.lstrip("#")
        r2, g2, b2 = (int(hx[i:i+2], 16) for i in (0, 2, 4))
        return f"#{int(r2*f):02x}{int(g2*f):02x}{int(b2*f):02x}"

    def draw(c2=color):
        cv.delete("all")
        cv.create_arc(0, 0, 2*r, h, start=90,  extent=180,  fill=c2, outline=c2)
        cv.create_arc(w-2*r, 0, w, h, start=270, extent=180, fill=c2, outline=c2)
        cv.create_rectangle(r, 0, w-r, h, fill=c2, outline=c2)
        cv.create_text(w//2, h//2, text=text, fill=C["white"],
                       font=("Helvetica", 9, "bold"))

    draw()
    cv.bind("<Enter>",    lambda _: draw(darken(color)))
    cv.bind("<Leave>",    lambda _: draw(color))
    cv.bind("<Button-1>", lambda _: (draw(darken(color, .70)),
                                     cv.after(120, lambda: draw(color)),
                                     cv.after(120, cmd)))
    return cv


def section_card(parent, title, accent, pady=6):
    """Card with a coloured left accent strip and title bar."""
    outer = tk.Frame(parent, bg=C["bg"])
    outer.pack(fill=tk.X, pady=pady)
    card = tk.Frame(outer, bg=C["card"],
                    highlightbackground=C["border"], highlightthickness=1)
    card.pack(fill=tk.X)
    # Accent strip
    tk.Frame(card, bg=accent, width=4).pack(side=tk.LEFT, fill=tk.Y)
    body = tk.Frame(card, bg=C["card"])
    body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    tk.Label(body, text=title, bg=accent, fg=C["white"],
             font=("Helvetica", 8, "bold"), padx=8, pady=4,
             anchor="w").pack(fill=tk.X)
    content = tk.Frame(body, bg=C["card"], padx=8, pady=6)
    content.pack(fill=tk.X)
    return content


def mini_stat(parent, label, accent, col, row):
    """Tactical telemetry tile with glowing value readout."""
    f = tk.Frame(parent, bg="#141414",
                 highlightbackground=accent, highlightthickness=1)
    f.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
    tk.Label(f, text=label.upper(), bg="#141414", fg=C["txt2"],
             font=("Helvetica", 7, "bold")).pack(anchor="w", padx=6, pady=(4, 0))
    val = tk.Label(f, text="—", bg="#141414", fg=accent,
                   font=("Courier", 11, "bold"))
    val.pack(anchor="w", padx=6, pady=(0, 4))
    return val


# ─── Main UI class ────────────────────────────────────────────────────────────

class RoutePlannerCoreUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 26 – AI Tourist Route Planner")
        self.root.configure(bg=C["bg"])
        self.root.resizable(False, False)

        # Backend
        self.env         = SpatialEnvironment(20, 20)
        self.csp         = StructuralConstraintEngine(baseline_financial_cap=400.0)
        self.uncertainty = BayesianUncertaintyEngine()
        self.decision    = SequentialDecisionMatrix()
        self.xai_engine  = ExplainablePipelineEngine(self.env, self.csp, self.uncertainty)

        self._running    = False
        self._anim_jobs  = []
        self._mode       = "OBSTACLE"   # current draw mode
        self._mode_btns  = {}           # mode_id -> Canvas widget
        self._sequential_routing = False

        self._build_header()

        # Three-column body
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self._build_left(body)
        self._build_middle(body)
        self._build_right(body)

        self._render_grid()
        self.display_canvas.bind("<Button-1>", self._lclick)
        self.display_canvas.bind("<Button-3>", self._rclick)

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C["header"], highlightbackground=C["border"], highlightthickness=1)
        hdr.pack(fill=tk.X)
        
        # Left branding
        brand_f = tk.Frame(hdr, bg=C["header"])
        brand_f.pack(side=tk.LEFT, padx=14, pady=8)
        tk.Label(brand_f, text="🗺  AEROPATH AI",
                 bg=C["header"], fg=C["secondary"],
                 font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        tk.Label(brand_f, text=" |  TACTICAL HUD v4.2  (TOURIST ROUTE PLANNER)",
                 bg=C["header"], fg=C["header_txt"],
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(4, 0))
                 
        # Right glowing status badge
        badge_f = tk.Frame(hdr, bg="#0A0A0A", highlightbackground=C["secondary"], highlightthickness=1, padx=10, pady=3)
        badge_f.pack(side=tk.RIGHT, padx=14, pady=8)
        self.status_lbl = tk.Label(badge_f, text="● READY",
                                   bg="#0A0A0A", fg=C["secondary"],
                                   font=("Helvetica", 9, "bold"))
        self.status_lbl.pack()

    # ── Left column: grid + landmarks ─────────────────────────────────────────

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=C["bg"])
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), pady=6)

        m2_hdr = tk.Frame(left, bg=C["m2"])
        m2_hdr.pack(fill=tk.X)
        tk.Label(m2_hdr, text="① Planning & Search Controls",
                 bg=C["m2"], fg=C["white"],
                 font=("Helvetica", 9, "bold"), pady=4, padx=8, anchor="w").pack(fill=tk.X)

        self._build_module2(left)
        self._build_module3(left)
        self._build_landmark_routing(left)

    def _build_landmark_routing(self, parent):
        route_card = tk.Frame(parent, bg=C["card"],
                              highlightbackground=C["border"], highlightthickness=1)
        route_card.pack(fill=tk.X, pady=(6, 0))
        tk.Label(route_card, text="📍 Tourist Landmark Routing",
                 bg=C["m1"], fg=C["white"],
                 font=("Helvetica", 8, "bold"), padx=8, pady=4, anchor="w").pack(fill=tk.X)

        rf = tk.Frame(route_card, bg=C["card"], padx=8, pady=6)
        rf.pack(fill=tk.X)

        lf = tk.Frame(rf, bg=C["card"])
        lf.pack(fill=tk.X, pady=(0, 4))
        self.curr_loc_lbl = tk.Label(lf, text="📍 Current: Beach 🏖 (0,0)", bg=C["card"], fg=C["m1"],
                                     font=("Helvetica", 8, "bold"), anchor="w")
        self.curr_loc_lbl.pack(fill=tk.X)

        df = tk.Frame(rf, bg=C["card"])
        df.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        tk.Label(df, text="🎯 Next Destination", bg=C["card"], fg=C["txt2"],
                 font=("Helvetica", 7, "bold")).pack(anchor="w", pady=(0, 2))
        self.goal_lm_var = tk.StringVar(value="Museum 🏛 (3,5)")
        self.goal_lm_cb = ttk.Combobox(df, textvariable=self.goal_lm_var, state="readonly", font=("Helvetica", 8), width=12)
        self.goal_lm_cb['values'] = ("Beach 🏖 (0,0)", "Museum 🏛 (3,5)", "Temple ⛩ (10,12)", "Park 🌳 (14,8)", "Mall 🛍 (19,19)")
        self.goal_lm_cb.pack(fill=tk.X)

        af = tk.Frame(rf, bg=C["card"])
        af.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        tk.Label(af, text="⚙ Algorithm", bg=C["card"], fg=C["txt2"],
                 font=("Helvetica", 7, "bold")).pack(anchor="w", pady=(0, 2))
        self.route_algo_var = tk.StringVar(value="A*")
        self.route_algo_cb = ttk.Combobox(af, textvariable=self.route_algo_var, state="readonly", font=("Helvetica", 8), width=8)
        self.route_algo_cb['values'] = ("A*", "BFS", "UCS", "DFS")
        self.route_algo_cb.pack(fill=tk.X)

        self.tour_mode_var = tk.BooleanVar(value=False)
        tk.Checkbutton(rf, text=" Stitched Sequential Tour", variable=self.tour_mode_var,
                       bg=C["card"], fg=C["txt"], selectcolor=C["bg"],
                       font=("Helvetica", 8, "bold"), activebackground=C["card"],
                       activeforeground=C["txt"]).pack(anchor="w", pady=(4, 0))

        btn_f = tk.Frame(route_card, bg=C["card"], padx=8, pady=4)
        btn_f.pack(fill=tk.X)

        def _route_landmarks():
            if self._running: return
            algo_id = self.route_algo_var.get()
            lm_coords = {
                "Beach 🏖 (0,0)": (0, 0),
                "Museum 🏛 (3,5)": (3, 5),
                "Temple ⛩ (10,12)": (10, 12),
                "Park 🌳 (14,8)": (14, 8),
                "Mall 🛍 (19,19)": (19, 19)
            }
            start_coord = self.env.start_coordinate

            if self.tour_mode_var.get():
                tour_sequence = [
                    (0, 0),    # Beach
                    (3, 5),    # Museum
                    (10, 12),  # Temple
                    (14, 8),   # Park
                    (19, 19)   # Mall
                ]
                milestones = [start_coord]
                for node in tour_sequence:
                    if node != milestones[-1]:
                        milestones.append(node)

                if len(milestones) < 2:
                    self._set_status("Already visited all landmarks!", C["goal"])
                    return

                self._running = True
                self._cancel_anims()
                self._render_grid()
                self._set_status(f"Running tour using {algo_id}…", "#FCD34D")

                t0 = time.perf_counter()
                full_path = []
                full_explored = []

                vf = self.csp.assess_cell_viability
                cf = self.uncertainty.calculate_dynamic_step_cost

                for leg_start, leg_end in zip(milestones[:-1], milestones[1:]):
                    self.env.reconfigure_start_state(*leg_start)
                    self.env.reconfigure_goal_state(*leg_end)
                    self.csp.remove_impassable_obstacle(*leg_end)
                    self.uncertainty.unregister_high_traffic_zone(*leg_end)

                    if   algo_id == "BFS": path, explored = search.execute_breadth_first_search(self.env, vf, cf)
                    elif algo_id == "DFS": path, explored = search.execute_depth_first_search(self.env, vf, cf)
                    elif algo_id == "UCS": path, explored = search.execute_uniform_cost_search(self.env, vf, cf)
                    else:                  path, explored = search.execute_astar_search(self.env, vf, cf)

                    if path:
                        if not full_path:
                            full_path.extend(path)
                        else:
                            full_path.extend(path[1:])
                    full_explored.extend(explored)

                elapsed = (time.perf_counter() - t0) * 1000

                self.env.reconfigure_start_state(*start_coord)
                final_goal = milestones[-1]
                self.env.reconfigure_goal_state(*final_goal)

                total_cost = self.uncertainty.calculate_accumulated_trajectory_cost(full_path)

                self.last_algo_id = algo_id
                self.last_path = full_path
                self.last_explored = full_explored
                self.last_elapsed = elapsed
                self.last_cost = total_cost

                self.sv_algo.config(text=algo_id + " Tour")
                self.sv_steps.config(text=str(len(full_path)))
                self.sv_nodes.config(text=str(len(full_explored)))
                self.sv_time.config(text=f"{elapsed:.1f}ms")

                compliant = self.csp.evaluate_resource_compliance(total_cost, elapsed / 1000)
                self.sv_cost.config(
                    text=f"Total Cost: {total_cost:.2f}  |  Budget OK: {'✓' if compliant else '✗'}",
                    fg=C["m2"] if compliant else C["m4"])
                self.csp_status.config(
                    text="✓ Tour within constraints" if compliant else "⚠ Tour exceeded constraints",
                    fg=C["m2"] if compliant else C["m4"])

                mm_val = self.decision.execute_minimax_lookahead(
                    self.env.start_coordinate, 3, True, self.env, cf)
                self.mm_score.config(text=f"{mm_val:.1f}")
                self.mm_best.config(text=algo_id)
                self.mm_trace.config(
                    text=f"Minimax evaluated depth-3 tree from Tour Start. Score={mm_val:.1f}.")

                prob, _, _ = self.uncertainty.evaluate_congestion_probability()
                self.b_storm.config(text=f"{self.uncertainty.probability_of_storm:.0%}")
                self.b_incident.config(text=f"{self.uncertainty.probability_of_road_incident:.0%}")
                self.b_congest.config(text=f"{prob:.0%}")
                self.b_delay.config(text=str(len(self.uncertainty.high_traffic_risk_cells)))

                self.route_trace.config(text=f"Tour: Start → Beach → Museum → Temple → Park → Mall")
                xai_desc = (
                    f"Planned a multi-landmark Tour visiting all 5 major tourist landmarks using {algo_id} search. "
                    f"Stitched sequential trajectory is {len(full_path)} cells long, expanding {len(full_explored)} states leg-by-leg. "
                    f"Total accumulated travel cost evaluates to {total_cost:.2f} resource units."
                )
                self.xai_text.config(state=tk.NORMAL)
                self.xai_text.delete("1.0", tk.END)
                self.xai_text.insert(tk.END, xai_desc)
                self.xai_text.config(state=tk.DISABLED)

                self._anim_explored(full_explored, full_path, algo_id)
            else:
                goal_coord = lm_coords[self.goal_lm_var.get()]
                if start_coord == goal_coord:
                    self._set_status("Already at destination!", C["goal"])
                    return
                self._sequential_routing = True
                self.csp.remove_impassable_obstacle(*goal_coord)
                self.env.reconfigure_goal_state(*goal_coord)
                self.csp.remove_impassable_obstacle(*goal_coord)
                self.uncertainty.unregister_high_traffic_zone(*goal_coord)
                self._render_grid()
                self._run(algo_id)

        self.route_landmarks_fn = _route_landmarks
        self.start_journey_btn = pill_btn(btn_f, "🚀 Start Journey", C["m2"],
                                          cmd=_route_landmarks, w=388, h=32)
        self.start_journey_btn.pack(pady=2)

    def _build_middle(self, parent):
        middle = tk.Frame(parent, bg=C["bg"])
        middle.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), pady=6)

        m1_hdr = tk.Frame(middle, bg=C["m1"])
        m1_hdr.pack(fill=tk.X)
        tk.Label(m1_hdr, text="② Interactive Map Space",
                 bg=C["m1"], fg=C["white"],
                 font=("Helvetica", 9, "bold"), pady=4, padx=8, anchor="w").pack(fill=tk.X)

        mode_card = tk.Frame(middle, bg=C["card"],
                             highlightbackground=C["border"], highlightthickness=1)
        mode_card.pack(fill=tk.X, pady=(0, 2))
        tk.Label(mode_card, text="✏  Click Tool Mode",
                 bg="#1E293B", fg=C["txt"],
                 font=("Helvetica", 8, "bold"), padx=8, pady=3, anchor="w").pack(fill=tk.X)

        mode_row = tk.Frame(mode_card, bg=C["card"], padx=6, pady=5)
        mode_row.pack(fill=tk.X)

        def _darken(hx, f=0.80):
            hx = hx.lstrip("#")
            r2,g2,b2 = (int(hx[i:i+2],16) for i in (0,2,4))
            return f"#{int(r2*f):02x}{int(g2*f):02x}{int(b2*f):02x}"

        for mode_id, (label, color) in MODES.items():
            cv = tk.Canvas(mode_row, width=82, height=28,
                           bg=C["card"], highlightthickness=0, cursor="hand2")
            cv.pack(side=tk.LEFT, padx=2)
            self._mode_btns[mode_id] = (cv, color)

            def _draw_btn(c2, cv=cv, label=label):
                cv.delete("all")
                cv.create_rectangle(2, 2, 80, 26, fill=c2, outline=c2, width=0)
                cv.create_text(41, 14, text=label, fill=C["white"],
                               font=("Helvetica", 8, "bold"))

            def _select(mid=mode_id):
                self._set_mode(mid)

            _draw_btn(color if mode_id != self._mode else _darken(color, 0.70))
            cv.bind("<Button-1>", lambda _, mid=mode_id: _select(mid))
            cv.bind("<Enter>",    lambda _, cv=cv, color=color: cv.itemconfig("all"))

        self.mode_indicator = tk.Label(mode_card,
            text="Mode: ⬛ Obstacle  |  Left-click on grid to paint",
            bg=C["card"], fg=C["txt2"], font=("Helvetica", 7), pady=3)
        self.mode_indicator.pack(fill=tk.X, padx=8)

        canvas_card = tk.Frame(middle, bg=C["card"],
                               highlightbackground=C["border"], highlightthickness=1)
        canvas_card.pack()
        grid_px = self.env.total_cols * CELL_PX
        self.display_canvas = tk.Canvas(canvas_card, width=grid_px, height=grid_px,
                                        bg=C["free"], highlightthickness=0)
        self.display_canvas.pack(padx=4, pady=4)

        type_card = tk.Frame(middle, bg=C["card"],
                             highlightbackground=C["border"], highlightthickness=1)
        type_card.pack(fill=tk.X, pady=(4, 0))
        tk.Label(type_card, text="Cell Colour Key",
                 bg="#1E293B", fg=C["txt"],
                 font=("Helvetica", 8, "bold"), padx=8, pady=3, anchor="w").pack(fill=tk.X)
        legend_data = [
            ("Start",     C["start"]),    ("Goal",      C["goal"]),
            ("Obstacle",  C["obstacle"]), ("Traffic",   C["traffic"]),
            ("Explored",  C["explored"]), ("Path",      C["path"]),
            ("Landmark",  C["landmark"]),
        ]
        lg = tk.Frame(type_card, bg=C["card"], padx=8, pady=4)
        lg.pack(fill=tk.X)
        for i, (lbl, col) in enumerate(legend_data):
            r, cv2 = divmod(i, 4)
            dot = tk.Canvas(lg, width=12, height=12, bg=C["card"], highlightthickness=0)
            dot.create_oval(1, 1, 11, 11, fill=col, outline=C["border"])
            dot.grid(row=r, column=cv2*2, padx=(2, 1), pady=1)
            tk.Label(lg, text=lbl, bg=C["card"], fg=C["txt"],
                     font=("Helvetica", 7)).grid(row=r, column=cv2*2+1, padx=(0, 8), sticky="w")
        self._update_curr_loc_lbl()

        self._build_help_card(middle)

    def _build_metrics(self, parent):
        c = section_card(parent, "② Search Performance Metrics", C["m5"], pady=4)
        sg = tk.Frame(c, bg=C["card"])
        sg.pack(fill=tk.X, pady=(2, 0))
        sg.columnconfigure(0, weight=1)
        sg.columnconfigure(1, weight=1)
        sg.columnconfigure(2, weight=1)
        sg.columnconfigure(3, weight=1)

        self.sv_algo  = mini_stat(sg, "Algorithm",     C["m5"], 0, 0)
        self.sv_steps = mini_stat(sg, "Path Steps",    C["m1"], 1, 0)
        self.sv_nodes = mini_stat(sg, "Nodes Expanded",C["m3"], 2, 0)
        self.sv_time  = mini_stat(sg, "Compute Time",  C["m2"], 3, 0)


    # ── Right column: all control modules ─────────────────────────────────────

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=C["bg"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=6)

        m6_hdr = tk.Frame(right, bg=C["header"])
        m6_hdr.pack(fill=tk.X)
        tk.Label(m6_hdr, text="③ AI Decision Support & Insights",
                 bg=C["header"], fg=C["white"],
                 font=("Helvetica", 9, "bold"), pady=4, padx=8, anchor="w").pack(fill=tk.X)

        self._build_metrics(right)
        self._build_module4(right)
        self._build_module5(right)
        self._build_module6(right)
        self._build_lab(right)

    # MODULE 2 – Search Algorithms ─────────────────────────────────────────────
    def _build_module2(self, parent):
        c = section_card(parent, "② Search Algorithms  (Path Finding)", C["m2"], pady=4)

        row1 = tk.Frame(c, bg=C["card"])
        row1.pack(fill=tk.X, pady=2)
        row2 = tk.Frame(c, bg=C["card"])
        row2.pack(fill=tk.X, pady=2)

        algos = [
            ("⬜ BFS  – Shortest Path",  "BFS",  C["m1"]),
            ("🔁 DFS  – Exploration",    "DFS",  C["m2"]),
            ("⚖  UCS  – Min Cost",       "UCS",  C["m3"]),
            ("⭐ A*   – Heuristic",      "A*",   C["m5"]),
        ]
        rows = [row1, row1, row2, row2]
        for i, (lbl, aid, col) in enumerate(algos):
            btn = pill_btn(rows[i], lbl, col,
                           cmd=lambda a=aid: self._run(a), w=188, h=32)
            btn.pack(side=tk.LEFT, padx=4)

        # Speed selection frame
        speed_f = tk.Frame(c, bg=C["card"])
        speed_f.pack(fill=tk.X, pady=(4, 2))
        tk.Label(speed_f, text="⚡ Animation Speed:", bg=C["card"], fg=C["txt"],
                 font=("Helvetica", 8, "bold")).pack(side=tk.LEFT, padx=(4, 6))
        self.speed_var = tk.StringVar(value="Normal")
        self.speed_cb = ttk.Combobox(speed_f, textvariable=self.speed_var, state="readonly", font=("Helvetica", 8), width=10)
        self.speed_cb['values'] = ("Instant", "Fast", "Normal", "Slow")
        self.speed_cb.pack(side=tk.LEFT)

        tk.Frame(c, bg=C["border"], height=1).pack(fill=tk.X, pady=(6, 2))
        pill_btn(c, "✕  Clear Grid", C["m4"],
                 cmd=self._clear, w=388, h=30).pack(pady=2)

        # Obstacle Density Slider & Button
        tk.Frame(c, bg=C["border"], height=1).pack(fill=tk.X, pady=(6, 2))
        obs_f = tk.Frame(c, bg=C["card"])
        obs_f.pack(fill=tk.X, pady=2)
        
        tk.Label(obs_f, text="🧱 Density:", bg=C["card"], fg=C["txt"],
                 font=("Helvetica", 8, "bold")).pack(side=tk.LEFT, padx=(4, 6))
        self.obstacle_density_var = tk.DoubleVar(value=0.20)
        self.obs_density_lbl = tk.Label(obs_f, text="20%", bg=C["card"], fg=C["m2"],
                                        font=("Helvetica", 8, "bold"))
        self.obs_density_lbl.pack(side=tk.LEFT, padx=(0, 6))
        
        sl_o = ttk.Scale(obs_f, from_=0.05, to=0.45, variable=self.obstacle_density_var,
                         orient=tk.HORIZONTAL, length=120,
                         command=lambda v: self.obs_density_lbl.config(text=f"{int(float(v)*100)}%"))
        sl_o.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        pill_btn(obs_f, "🎲 Populate", C["m2"],
                 cmd=self._generate_random_obstacles, w=100, h=28).pack(side=tk.RIGHT)

    # MODULE 3 – CSP ───────────────────────────────────────────────────────────
    def _build_module3(self, parent):
        c = section_card(parent, "③ Constraint Satisfaction Problem (CSP Engine)", C["m3"], pady=4)

        row = tk.Frame(c, bg=C["card"])
        row.pack(fill=tk.X)

        # Budget slider
        bf = tk.Frame(row, bg=C["card"])
        bf.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 8))
        tk.Label(bf, text="💰 Budget Limit", bg=C["card"], fg=C["txt"],
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        self.budget_var = tk.DoubleVar(value=400.0)
        self.budget_lbl = tk.Label(bf, text="₹ 400", bg=C["card"], fg=C["m3"],
                                   font=("Helvetica", 11, "bold"))
        self.budget_lbl.pack(anchor="w")
        sl_b = ttk.Scale(bf, from_=100, to=1000, variable=self.budget_var,
                         orient=tk.HORIZONTAL, length=170,
                         command=lambda v: (
                             self.budget_lbl.config(text=f"₹ {int(float(v))}"),
                             setattr(self.csp, 'maximum_financial_budget', float(v))
                         ))
        sl_b.pack(fill=tk.X)

        # Time slider
        tf = tk.Frame(row, bg=C["card"])
        tf.pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Label(tf, text="⏱ Time Limit", bg=C["card"], fg=C["txt"],
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        self.time_var = tk.DoubleVar(value=120.0)
        self.time_lbl = tk.Label(tf, text="120 min", bg=C["card"], fg=C["m3"],
                                 font=("Helvetica", 11, "bold"))
        self.time_lbl.pack(anchor="w")
        sl_t = ttk.Scale(tf, from_=30, to=300, variable=self.time_var,
                         orient=tk.HORIZONTAL, length=170,
                         command=lambda v: (
                             self.time_lbl.config(text=f"{int(float(v))} min"),
                             setattr(self.csp, 'maximum_time_limit', float(v))
                         ))
        sl_t.pack(fill=tk.X)

        # CSP status
        tk.Frame(c, bg=C["border"], height=1).pack(fill=tk.X, pady=4)
        sf = tk.Frame(c, bg=C["card"])
        sf.pack(fill=tk.X)
        tk.Label(sf, text="Forward Checking:", bg=C["card"], fg=C["txt2"],
                 font=("Helvetica", 8)).pack(side=tk.LEFT)
        self.csp_status = tk.Label(sf, text="Eliminating invalid paths…",
                                   bg=C["card"], fg=C["m3"],
                                   font=("Helvetica", 8, "bold"))
        self.csp_status.pack(side=tk.LEFT, padx=6)

        self.sv_cost = tk.Label(c, text="Total Cost: —", bg=C["card"],
                                fg=C["txt"], font=("Helvetica", 9, "bold"))
        self.sv_cost.pack(anchor="w", pady=(4, 0))

    # MODULE 4 – Decision Making ───────────────────────────────────────────────
    def _build_module4(self, parent):
        c = section_card(parent, "④ Decision Making", C["m4"], pady=4)
        tk.Label(c, text="Minimax Algorithm", bg=C["card"], fg=C["txt"],
                 font=("Helvetica", 8, "bold")).pack(anchor="w")

        sf = tk.Frame(c, bg=C["card"])
        sf.pack(fill=tk.X, pady=2)
        sf.columnconfigure(0, weight=1)
        sf.columnconfigure(1, weight=1)
        self.mm_score = mini_stat(sf, "Minimax Score", C["m4"], 0, 0)
        self.mm_best  = mini_stat(sf, "Best Algorithm", C["m5"], 1, 0)

        self.mm_trace = tk.Label(c, text="Awaiting run…", bg=C["card"],
                                 fg=C["txt2"], font=("Helvetica", 7),
                                 wraplength=190, justify=tk.LEFT)
        self.mm_trace.pack(anchor="w", pady=(4, 0))

    # MODULE 5 – Uncertainty ───────────────────────────────────────────────────
    def _build_module5(self, parent):
        c = section_card(parent, "⑤ Reasoning Under Uncertainty (Bayesian Network)", C["m5"], pady=4)

        bf = tk.Frame(c, bg=C["card"])
        bf.pack(fill=tk.X, pady=(0, 4))
        bf.columnconfigure(0, weight=1)
        bf.columnconfigure(1, weight=1)
        self.b_storm    = mini_stat(bf, "P(Storm)",     C["m5"], 0, 0)
        self.b_incident = mini_stat(bf, "P(Incident)",  C["m5"], 1, 0)
        self.b_congest  = mini_stat(bf, "P(Congestion)",C["m5"], 0, 1)
        self.b_delay    = mini_stat(bf, "Traffic Cells", C["m3"], 1, 1)

        self.b_storm.config(text=f"{self.uncertainty.probability_of_storm:.0%}")
        self.b_incident.config(text=f"{self.uncertainty.probability_of_road_incident:.0%}")
        self.b_congest.config(text="—")
        self.b_delay.config(text="0")

        # Dynamic parameter tuner sliders
        row = tk.Frame(c, bg=C["card"])
        row.pack(fill=tk.X, pady=4)
        
        # Storm slider
        sf = tk.Frame(row, bg=C["card"])
        sf.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 8))
        tk.Label(sf, text="⚡ Storm Prior", bg=C["card"], fg=C["txt"],
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        self.storm_var = tk.DoubleVar(value=0.30)
        self.storm_lbl = tk.Label(sf, text="30%", bg=C["card"], fg=C["m5"],
                                  font=("Helvetica", 10, "bold"))
        self.storm_lbl.pack(anchor="w")
        sl_s = ttk.Scale(sf, from_=0.0, to=1.0, variable=self.storm_var,
                         orient=tk.HORIZONTAL, length=170,
                         command=lambda v: (
                             self.storm_lbl.config(text=f"{int(float(v)*100)}%"),
                             setattr(self.uncertainty, 'probability_of_storm', float(v)),
                             self._update_bayes_inference()
                         ))
        sl_s.pack(fill=tk.X)

        # Incident slider
        if_ = tk.Frame(row, bg=C["card"])
        if_.pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Label(if_, text="🚗 Incident Prior", bg=C["card"], fg=C["txt"],
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        self.incident_var = tk.DoubleVar(value=0.15)
        self.incident_lbl = tk.Label(if_, text="15%", bg=C["card"], fg=C["m5"],
                                     font=("Helvetica", 10, "bold"))
        self.incident_lbl.pack(anchor="w")
        sl_i = ttk.Scale(if_, from_=0.0, to=1.0, variable=self.incident_var,
                         orient=tk.HORIZONTAL, length=170,
                         command=lambda v: (
                             self.incident_lbl.config(text=f"{int(float(v)*100)}%"),
                             setattr(self.uncertainty, 'probability_of_road_incident', float(v)),
                             self._update_bayes_inference()
                         ))
        sl_i.pack(fill=tk.X)

        pill_btn(c, "🔄 Randomise Priors", C["m5"],
                 cmd=self._refresh_bayes, w=388, h=30).pack(pady=(6, 0))

    # MODULE 6 – Integrated Pipeline + XAI ────────────────────────────────────
    def _build_module6(self, parent):
        c = section_card(parent, "⑥ Tactical XAI Console & Reasoning Stream", C["m6"], pady=4)

        # Route trace row
        self.route_trace = tk.Label(c, text="Route: —", bg=C["card"],
                                    fg=C["secondary"], font=("Helvetica", 9, "bold"),
                                    wraplength=420, justify=tk.LEFT)
        self.route_trace.pack(anchor="w", pady=(0, 4))

        # Terminal Window Container
        term_frame = tk.Frame(c, bg="#0A0A0A", highlightbackground=C["border"], highlightthickness=1)
        term_frame.pack(fill=tk.X)

        # Terminal Top Bar
        tb = tk.Frame(term_frame, bg="#1E1E1E", height=18)
        tb.pack(fill=tk.X)
        tk.Label(tb, text="● ● ●   >_ XAI_TELEMETRY_STREAM", bg="#1E1E1E", fg="#94A3B8",
                 font=("Courier", 7, "bold"), padx=6).pack(side=tk.LEFT)

        self.xai_text = tk.Text(term_frame, width=58, height=6,
                                font=("Courier", 8),
                                bg="#0A0A0A", fg="#E2E8F0",
                                relief="flat", wrap=tk.WORD,
                                padx=6, pady=4,
                                insertbackground=C["secondary"])
        self.xai_text.pack(fill=tk.X)
        self.xai_text.insert(tk.END,
            "> SYSTEM_READY: AeroPath Tactical HUD Initialized.\n"
            "> [PROTOCOL]: Select search algorithm to stream reasoning.\n"
            "> [XAI]: Real-time CSP bounds & Bayesian risk telemetry active.")
        self.xai_text.config(state=tk.DISABLED)

        pill_btn(c, "💡 Why This Route? — AI Analysis", C["m6"],
                 cmd=self._show_xai_details, w=388, h=30).pack(pady=(6, 0))

    def _show_xai_details(self):
        if not hasattr(self, 'last_path') or not self.last_path:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "Please run a search algorithm first to evaluate a route!")
            return

        pop = tk.Toplevel(self.root)
        pop.title("Why This Route? — AI Explanation")
        pop.configure(bg=C["bg"])
        pop.resizable(False, False)
        
        hdr = tk.Frame(pop, bg=C["header"])
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="💡 WHY THIS ROUTE WAS SELECTED", bg=C["header"], fg=C["white"],
                 font=("Helvetica", 10, "bold"), pady=8, padx=12).pack(anchor="w")

        body = tk.Frame(pop, bg=C["card"], padx=14, pady=12,
                        highlightbackground=C["border"], highlightthickness=1)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        algo = self.last_algo_id
        cost = self.last_cost
        steps = len(self.last_path)
        nodes = len(self.last_explored)
        elapsed = self.last_elapsed
        
        budget_ok = cost <= self.csp.maximum_financial_budget
        time_ok = (elapsed / 1000) <= self.csp.maximum_time_limit
        
        congested_crossed = sum(1 for n in self.last_path if n in self.uncertainty.high_traffic_risk_cells)
        obstacles_count = len(self.csp.structural_obstacles_set)
        prob, _, _ = self.uncertainty.evaluate_congestion_probability()

        rows = [
            ("🔍 1. Search Algorithm:", f"{algo}"),
            ("💰 2. Calculated Route Cost:", f"₹ {cost:.2f}"),
            ("🚶 3. Total Path Steps:", f"{steps} cells"),
            ("📈 4. State Space Explored:", f"{nodes} nodes expanded"),
            ("⏱ 5. Compute Latency:", f"{elapsed:.1f} ms"),
            ("⚖ 6. Budget Constraint Check:", "✓ Compliant" if budget_ok else "⚠ Exceeded budget"),
            ("⏱ 7. Time Constraint Check:", "✓ Compliant" if time_ok else "⚠ Exceeded time limit"),
            ("🚗 8. Congested Intersections:", f"{congested_crossed} zones crossed"),
            ("🧱 9. Obstacles Bypassed:", f"{obstacles_count} environmental obstacles avoided"),
            ("🟡 10. Bayesian Congestion Risk:", f"{prob:.0%}"),
        ]

        for label, val in rows:
            f = tk.Frame(body, bg=C["card"])
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, bg=C["card"], fg=C["txt2"], font=("Helvetica", 8, "bold")).pack(side=tk.LEFT)
            tk.Label(f, text=val, bg=C["card"], fg=C["txt"], font=("Helvetica", 8)).pack(side=tk.RIGHT)

        tk.Frame(body, bg=C["border"], height=1).pack(fill=tk.X, pady=8)
        summary = self.xai_engine.construct_natural_language_explanation(algo, self.last_path, nodes, cost)
        tk.Label(body, text="📝 AI Reasoning Summary:", bg=C["card"], fg=C["header"],
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        st = tk.Label(body, text=summary, bg=C["card"], fg=C["txt"], font=("Helvetica", 8),
                      wraplength=380, justify=tk.LEFT)
        st.pack(anchor="w", pady=(4, 0))

        pill_btn(body, "Close Window", C["m1"], cmd=pop.destroy, w=380, h=30).pack(pady=(12, 0))

    def _build_help_card(self, parent):
        c = section_card(parent, "ℹ  Algorithm Guide & Help", C["m1"], pady=4)
        info_text = (
            "🔍 Classical Search:\n"
            "  • BFS: Guarantees shortest path under unit cost.\n"
            "  • DFS: Explores deeply; paths may be sub-optimal.\n"
            "  • UCS: Evaluates cost g(n); optimal for variable steps.\n"
            "  • A*: Explores via f(n) = g(n) + h(n); fastest optimal path.\n\n"
            "⚖  CSP constraints:\n"
            "  • Filter paths exceeding budget & time limits.\n\n"
            "🤖 Minimax Strategy:\n"
            "  • Simulates dynamic traffic increases to verify durability.\n\n"
            "🟡 Bayesian Uncertainty:\n"
            "  • Estimates congestion risk based on storm & road priors."
        )
        lbl = tk.Label(c, text=info_text, bg=C["card"], fg=C["txt2"],
                       font=("Helvetica", 8), justify=tk.LEFT, anchor="w")
        lbl.pack(fill=tk.X, padx=4, pady=2)

    def _build_lab(self, parent):
        c = section_card(parent, "🧪 Algorithm Laboratory  (Performance Comparison)", C["m1"], pady=4)

        # Comparison triggers
        btn_f = tk.Frame(c, bg=C["card"])
        btn_f.pack(fill=tk.X, pady=(2, 6))

        pill_btn(btn_f, "🧪 Run Comparison Analysis", C["m1"],
                 cmd=self._run_lab_comparison, w=388, h=30).pack(pady=2)
        pill_btn(btn_f, "💾 Export Comparative Analysis", C["m2"],
                 cmd=self._export_analysis, w=388, h=30).pack(pady=2)

        # Tabular structure using a Frame with labels
        self.lab_table_frame = tk.Frame(c, bg=C["border"])
        self.lab_table_frame.pack(fill=tk.X, pady=2)

        # Configure columns
        self.lab_table_frame.columnconfigure(0, weight=2) # Algorithm name
        self.lab_table_frame.columnconfigure(1, weight=1) # Path Cost
        self.lab_table_frame.columnconfigure(2, weight=1) # Nodes Exp
        self.lab_table_frame.columnconfigure(3, weight=1) # Time (ms)
        self.lab_table_frame.columnconfigure(4, weight=1) # Status

        # Headers
        headers = ["Algorithm", "Cost", "Nodes", "Time", "Status"]
        for col_idx, text in enumerate(headers):
            lbl = tk.Label(self.lab_table_frame, text=text, bg=C["card"], fg=C["txt"],
                           font=("Helvetica", 8, "bold"), bd=1, relief=tk.FLAT, pady=4)
            lbl.grid(row=0, column=col_idx, sticky="ew", padx=1, pady=1)

        # Row entries placeholders
        self.lab_rows = {}
        algos = [("BFS", "BFS"), ("DFS", "DFS"), ("UCS", "UCS"), ("A*", "A*")]
        for row_idx, (name, aid) in enumerate(algos, start=1):
            row_widgets = []
            # Name label
            lbl_name = tk.Label(self.lab_table_frame, text=name, bg=C["bg"], fg=C["txt"],
                                font=("Helvetica", 8, "bold"), anchor="w", padx=6, pady=4)
            lbl_name.grid(row=row_idx, column=0, sticky="ew", padx=1, pady=1)
            row_widgets.append(lbl_name)

            # Metric labels: Cost, Nodes, Time, Status
            for col_idx in range(1, 5):
                lbl_metric = tk.Label(self.lab_table_frame, text="-", bg=C["bg"], fg=C["txt"],
                                      font=("Helvetica", 8), anchor="center", pady=4)
                lbl_metric.grid(row=row_idx, column=col_idx, sticky="ew", padx=1, pady=1)
                row_widgets.append(lbl_metric)

            self.lab_rows[aid] = row_widgets

    def _run_lab_comparison(self):
        """
        Runs all four search algorithms sequentially on the current grid
        without animation, collects metrics, and displays them in the table.
        """
        valid, err_msg = self._validate_planning_state()
        if not valid:
            self._set_status(f"Lab: {err_msg}", C["goal"])
            return

        self._set_status("Running Lab Comparison...", "#FCD34D")
        self.root.update_idletasks()

        vf = self.csp.assess_cell_viability
        cf = self.uncertainty.calculate_dynamic_step_cost

        # Save comparison data for export
        self.comparison_results = {}

        algos = [
            ("BFS", search.execute_breadth_first_search),
            ("DFS", search.execute_depth_first_search),
            ("UCS", search.execute_uniform_cost_search),
            ("A*", search.execute_astar_search)
        ]

        for aid, search_fn in algos:
            t0 = time.perf_counter()
            path, explored = search_fn(self.env, vf, cf)
            elapsed = (time.perf_counter() - t0) * 1000  # ms

            cost = self.uncertainty.calculate_accumulated_trajectory_cost(path)

            row_widgets = self.lab_rows[aid]

            if path:
                row_widgets[1].config(text=f"{cost:.1f}")
                row_widgets[2].config(text=str(len(explored)))
                row_widgets[3].config(text=f"{elapsed:.2f}ms")
                row_widgets[4].config(text="✓ Found", fg="#6EE7B7")
                self.comparison_results[aid] = {
                    "cost": f"{cost:.1f}",
                    "nodes": len(explored),
                    "time": f"{elapsed:.2f}ms",
                    "status": "Found"
                }
            else:
                row_widgets[1].config(text="-")
                row_widgets[2].config(text=str(len(explored)))
                row_widgets[3].config(text=f"{elapsed:.2f}ms")
                row_widgets[4].config(text="✗ No Path", fg=C["goal"])
                self.comparison_results[aid] = {
                    "cost": "-",
                    "nodes": len(explored),
                    "time": f"{elapsed:.2f}ms",
                    "status": "No Path"
                }

        self._set_status("Lab analysis complete ✓", "#6EE7B7")

    def _export_analysis(self):
        """
        Exports the current planning scenario, comparative laboratory metrics,
        and XAI trace to a portfolio-ready text file 'route_analysis.txt'.
        """
        # Run comparison if it hasn't been run yet
        if not hasattr(self, 'comparison_results') or not self.comparison_results:
            self._run_lab_comparison()
            if not hasattr(self, 'comparison_results') or not self.comparison_results:
                # Validation failed in _run_lab_comparison
                return

        try:
            with open("route_analysis.txt", "w", encoding="utf-8") as f:
                f.write("====================================================\n")
                f.write("      AI TOURIST ROUTE PLANNER - ANALYSIS REPORT    \n")
                f.write("====================================================\n\n")

                start = self.env.start_coordinate
                goal = self.env.goal_coordinate
                f.write(f"Start Coordinate:       {start}\n")
                f.write(f"Goal Coordinate:        {goal}\n")
                f.write(f"Grid Configuration:     {self.env.total_rows}x{self.env.total_cols} Grid\n\n")

                f.write("----------------------------------------------------\n")
                f.write(" 1. PERFORMANCE COMPARISON MATRIX                   \n")
                f.write("----------------------------------------------------\n")
                f.write(f"{'Algorithm':<12} | {'Cost':<8} | {'Nodes Exp':<10} | {'Time (ms)':<10} | {'Status':<10}\n")
                f.write("-" * 55 + "\n")

                for aid in ["BFS", "DFS", "UCS", "A*"]:
                    res = self.comparison_results[aid]
                    f.write(f"{aid:<12} | {res['cost']:<8} | {res['nodes']:<10} | {res['time']:<10} | {res['status']:<10}\n")
                f.write("\n")

                f.write("----------------------------------------------------\n")
                f.write(" 2. DYNAMIC ENVIRONMENT CONSTRAINTS                 \n")
                f.write("----------------------------------------------------\n")
                f.write(f"Budget Limit:           ₹ {self.csp.maximum_financial_budget}\n")
                f.write(f"Travel Time Limit:      {self.csp.maximum_time_limit} min\n")

                # Count obstacles and traffic zones
                obs_count = sum(1 for r in range(self.env.total_rows) for c in range(self.env.total_cols) if not self.csp.assess_cell_viability((r,c)))
                f.write(f"Active Obstacles:       {obs_count}\n\n")

                f.write("----------------------------------------------------\n")
                f.write(" 3. EXPLAINABLE AI (XAI) DECISION TRACE             \n")
                f.write("----------------------------------------------------\n")
                xai_msg = self.xai_text.get("1.0", tk.END).strip()
                f.write(f"{xai_msg}\n\n")

                f.write("====================================================\n")
                f.write("Report Generated Successfully.\n")
                f.write("====================================================\n")

            self._set_status("Analysis exported to route_analysis.txt ✓", "#6EE7B7")
        except Exception as e:
            self._set_status(f"Export failed: {str(e)}", C["goal"])

    # ── Grid Rendering ─────────────────────────────────────────────────────────

    def _render_grid(self, overlay=None):
        overlay = overlay or {}
        self.display_canvas.delete("all")
        for r in range(self.env.total_rows):
            for c in range(self.env.total_cols):
                coord = (r, c)
                x0, y0 = c * CELL_PX, r * CELL_PX
                x1, y1 = x0 + CELL_PX, y0 + CELL_PX

                if coord in overlay:
                    fill = overlay[coord]
                elif coord == self.env.start_coordinate:
                    fill = C["start"]
                elif coord == self.env.goal_coordinate:
                    fill = C["goal"]
                elif not self.csp.assess_cell_viability(coord):
                    fill = C["obstacle"]
                elif coord in self.uncertainty.high_traffic_risk_cells:
                    fill = C["traffic"]
                elif coord in LANDMARKS:
                    fill = C["landmark"]
                else:
                    fill = C["free"]

                self.display_canvas.create_rectangle(
                    x0, y0, x1, y1, fill=fill,
                    outline=C["border"], width=0.5)

        # Draw landmark emoji labels
        for coord, (name, emoji) in LANDMARKS.items():
            if coord == self.env.start_coordinate or coord == self.env.goal_coordinate:
                continue
            if not self.csp.assess_cell_viability(coord):
                continue
            r, c = coord
            self.display_canvas.create_text(
                c * CELL_PX + CELL_PX // 2,
                r * CELL_PX + CELL_PX // 2,
                text=emoji, font=("Helvetica", 9))

        self._stamp_icon(self.env.start_coordinate, "S", C["white"])
        self._stamp_icon(self.env.goal_coordinate,  "G", C["white"])

    def _stamp_icon(self, coord, text, fg):
        r, c = coord
        self.display_canvas.create_text(
            c * CELL_PX + CELL_PX // 2,
            r * CELL_PX + CELL_PX // 2,
            text=text, fill=fg, font=("Helvetica", 7, "bold"))

    def _paint_cell(self, coord, fill):
        r, c = coord
        x0, y0 = c * CELL_PX, r * CELL_PX
        self.display_canvas.create_rectangle(
            x0, y0, x0 + CELL_PX, y0 + CELL_PX,
            fill=fill, outline=C["border"], width=0.5)

    def _paint_robot(self, coord):
        r, c = coord
        x0, y0 = c * CELL_PX + 2, r * CELL_PX + 2
        self.display_canvas.create_oval(
            x0, y0, x0 + CELL_PX - 4, y0 + CELL_PX - 4,
            fill=C["robot"], outline=C["white"], width=1.5, tags="robot")

    # ── Mode helpers ────────────────────────────────────────────────────────────

    def _set_mode(self, mode_id):
        self._mode = mode_id
        label, color = MODES[mode_id]
        self.mode_indicator.config(
            text=f"Mode: {label}  |  Left-click on grid to paint")
        self._refresh_mode_buttons()
        self._set_status(f"Mode: {label}", "#FCD34D")

    def _refresh_mode_buttons(self):
        def _darken(hx, f=0.70):
            hx = hx.lstrip("#")
            r2,g2,b2 = (int(hx[i:i+2],16) for i in (0,2,4))
            return f"#{int(r2*f):02x}{int(g2*f):02x}{int(b2*f):02x}"

        for mid, (cv, color) in self._mode_btns.items():
            label = MODES[mid][0]
            active = (mid == self._mode)
            fill = _darken(color) if active else color
            cv.delete("all")
            # Sunken effect for active
            if active:
                cv.create_rectangle(2, 2, 80, 26, fill=fill,
                                    outline="#1E293B", width=2)
            else:
                cv.create_rectangle(2, 2, 80, 26, fill=fill,
                                    outline=fill, width=0)
            cv.create_text(41, 14, text=label, fill=C["white"],
                           font=("Helvetica", 8, "bold"))

    # ── Clicks ─────────────────────────────────────────────────────────────────

    def _lclick(self, e):
        if self._running: return
        col = e.x // CELL_PX
        row = e.y // CELL_PX
        # Bounds check
        if not (0 <= row < self.env.total_rows and 0 <= col < self.env.total_cols):
            return
        coord = (row, col)

        if self._mode == "SET_START":
            # Clear old start from obstacles/traffic just in case
            self.csp.remove_impassable_obstacle(*self.env.start_coordinate)
            self.env.reconfigure_start_state(row, col)
            # Remove any obstacle/traffic on new start
            self.csp.remove_impassable_obstacle(row, col)
            self.uncertainty.unregister_high_traffic_zone(row, col)
            self._set_status(f"Start set → ({row},{col})", C["start"])
            self._update_curr_loc_lbl()

        elif self._mode == "SET_GOAL":
            self.env.reconfigure_goal_state(row, col)
            self.csp.remove_impassable_obstacle(row, col)
            self.uncertainty.unregister_high_traffic_zone(row, col)
            self._set_status(f"Goal set → ({row},{col})", C["goal"])

        elif self._mode == "OBSTACLE":
            if coord in (self.env.start_coordinate, self.env.goal_coordinate): return
            if coord in self.csp.structural_obstacles_set:
                self.csp.remove_impassable_obstacle(row, col)
            else:
                self.csp.register_impassable_obstacle(row, col)
                self.uncertainty.unregister_high_traffic_zone(row, col)

        elif self._mode == "TRAFFIC":
            if coord in (self.env.start_coordinate, self.env.goal_coordinate): return
            if coord in self.uncertainty.high_traffic_risk_cells:
                self.uncertainty.unregister_high_traffic_zone(row, col)
            else:
                self.uncertainty.register_high_traffic_zone(row, col)
                self.csp.remove_impassable_obstacle(row, col)
            self.b_delay.config(text=str(len(self.uncertainty.high_traffic_risk_cells)))

        elif self._mode == "ERASE":
            if coord in (self.env.start_coordinate, self.env.goal_coordinate): return
            self.csp.remove_impassable_obstacle(row, col)
            self.uncertainty.unregister_high_traffic_zone(row, col)

        self._render_grid()

    def _rclick(self, e):
        """Right-click always quick-toggles traffic regardless of mode."""
        if self._running: return
        col = e.x // CELL_PX
        row = e.y // CELL_PX
        coord = (row, col)
        if coord in (self.env.start_coordinate, self.env.goal_coordinate): return
        if coord in self.uncertainty.high_traffic_risk_cells:
            self.uncertainty.unregister_high_traffic_zone(row, col)
        else:
            self.uncertainty.register_high_traffic_zone(row, col)
            self.csp.remove_impassable_obstacle(row, col)
        self._render_grid()
        self.b_delay.config(text=str(len(self.uncertainty.high_traffic_risk_cells)))

    # ── Algorithm Run & Animation ──────────────────────────────────────────────

    def _set_status(self, msg, color=None):
        if color is None or color == "#6EE7B7":
            color = C["secondary"]
        self.status_lbl.config(text=f"● {msg.upper()}", fg=color)

    def _validate_planning_state(self):
        """
        Validates start, goal, and boundary conditions.
        Returns (True, None) if valid, or (False, error_message) if invalid.
        """
        start = self.env.start_coordinate
        goal = self.env.goal_coordinate

        # 1. Bounds check
        for r, c in (start, goal):
            if not (0 <= r < self.env.total_rows and 0 <= c < self.env.total_cols):
                return False, f"Coordinates ({r},{c}) out of bounds!"

        # 2. Start == Goal check
        if start == goal:
            return False, "Already at destination!"

        # 3. Obstacle checks
        if not self.csp.assess_cell_viability(start):
            return False, "Start blocked by obstacle!"

        if not self.csp.assess_cell_viability(goal):
            return False, "Goal blocked by obstacle!"

        return True, None

    def _run(self, algo_id):
        if self._running: return

        valid, err_msg = self._validate_planning_state()
        if not valid:
            self._set_status(err_msg, C["goal"])
            self.csp_status.config(text=f"⚠ {err_msg}", fg=C["goal"])
            return

        self._running = True
        self._cancel_anims()
        self._render_grid()
        self._set_status(f"Running {algo_id}…", "#FCD34D")
        self.csp_status.config(text="Checking constraints…", fg=C["m3"])

        t0 = time.perf_counter()
        vf = self.csp.assess_cell_viability
        cf = self.uncertainty.calculate_dynamic_step_cost

        if   algo_id == "BFS": path, explored = search.execute_breadth_first_search(self.env, vf, cf)
        elif algo_id == "DFS": path, explored = search.execute_depth_first_search(self.env, vf, cf)
        elif algo_id == "UCS": path, explored = search.execute_uniform_cost_search(self.env, vf, cf)
        else:                  path, explored = search.execute_astar_search(self.env, vf, cf)

        elapsed = (time.perf_counter() - t0) * 1000
        total_cost = self.uncertainty.calculate_accumulated_trajectory_cost(path)

        self.last_algo_id = algo_id
        self.last_path = path
        self.last_explored = explored
        self.last_elapsed = elapsed
        self.last_cost = total_cost

        # ── Module 2 stats ──
        self.sv_algo.config(text=algo_id)
        self.sv_steps.config(text=str(len(path)))
        self.sv_nodes.config(text=str(len(explored)))
        self.sv_time.config(text=f"{elapsed:.1f}ms")

        # ── Module 3 CSP ──
        compliant = self.csp.evaluate_resource_compliance(total_cost, elapsed / 1000)
        self.sv_cost.config(
            text=f"Total Cost: {total_cost:.2f}  |  Budget OK: {'✓' if compliant else '✗'}",
            fg=C["m2"] if compliant else C["m4"])
        self.csp_status.config(
            text="✓ Path within constraints" if compliant else "⚠ Budget/time exceeded",
            fg=C["m2"] if compliant else C["m4"])

        # ── Module 4 Decision ──
        mm_val = self.decision.execute_minimax_lookahead(
            self.env.start_coordinate, 3, True, self.env, cf)
        self.mm_score.config(text=f"{mm_val:.1f}")
        self.mm_best.config(text=algo_id)
        self.mm_trace.config(
            text=f"Minimax evaluated depth-3 tree from Start. "
                 f"Best route minimises delay: score={mm_val:.1f}.")

        # ── Module 5 Uncertainty ──
        prob, storm, incident = self.uncertainty.evaluate_congestion_probability()
        self.b_storm.config(text=f"{self.uncertainty.probability_of_storm:.0%}")
        self.b_incident.config(text=f"{self.uncertainty.probability_of_road_incident:.0%}")
        self.b_congest.config(text=f"{prob:.0%}")
        self.b_delay.config(text=str(len(self.uncertainty.high_traffic_risk_cells)))

        # ── Module 6 XAI ──
        landmark_path = self._landmark_trace(path)
        self.route_trace.config(
            text=f"Route: {landmark_path}" if landmark_path else "Route: Start → … → Goal")
        xai = self.xai_engine.construct_natural_language_explanation(
            algo_id, path, len(explored), total_cost)
        formatted_log = (
            f"> [PROTOCOL]: {algo_id}_SEARCH EXECUTION\n"
            f"> [METRICS]: Steps={len(path)} | Explored={len(explored)} | Cost={total_cost:.2f}\n"
            f"> [REASONING]: {xai}"
        )
        self.xai_text.config(state=tk.NORMAL)
        self.xai_text.delete("1.0", tk.END)
        self.xai_text.insert(tk.END, formatted_log)
        self.xai_text.config(state=tk.DISABLED)

        # ── Animate ──
        self._anim_explored(explored, path, algo_id)

    def _update_curr_loc_lbl(self):
        coord = self.env.start_coordinate
        name = "Custom Point"
        emoji = "📍"
        for k, v in LANDMARKS.items():
            if k == coord:
                name, emoji = v[0], v[1]
                break
        if hasattr(self, 'curr_loc_lbl'):
            self.curr_loc_lbl.config(text=f"📍 Current: {name} {emoji} ({coord[0]},{coord[1]})")

    def _landmark_trace(self, path):
        """Build a human-readable landmark chain from the path."""
        visited_names = []
        for coord in path:
            if coord in LANDMARKS:
                visited_names.append(LANDMARKS[coord][0])
            elif coord == self.env.start_coordinate:
                visited_names.insert(0, "Start")
            elif coord == self.env.goal_coordinate:
                visited_names.append("Goal")
        # Deduplicate consecutive
        deduped = []
        for n in visited_names:
            if not deduped or deduped[-1] != n:
                deduped.append(n)
        return "  →  ".join(deduped) if deduped else ""

    # ── Animation steps ────────────────────────────────────────────────────────

    def _get_animation_delay(self, anim_type):
        """
        Returns the appropriate milliseconds delay based on selected speed.
        anim_type: 'explore', 'path', or 'robot'
        """
        if not hasattr(self, 'speed_var'):
            base_delays = {"explore": 6, "path": 28, "robot": 40}
            return base_delays.get(anim_type, 10)

        speed = self.speed_var.get()
        if speed == "Instant":
            return 0

        # Speed factor multipliers:
        # Fast: 0.25x of normal
        # Normal: 1.0x of normal
        # Slow: 3.0x of normal
        factors = {
            "Fast": 0.25,
            "Normal": 1.0,
            "Slow": 3.0
        }
        factor = factors.get(speed, 1.0)

        base_delays = {
            "explore": 6,
            "path": 28,
            "robot": 40
        }
        return int(base_delays.get(anim_type, 10) * factor)

    def _anim_explored(self, explored, path, algo_id):
        skip = {self.env.start_coordinate, self.env.goal_coordinate}
        nodes = [n for n in explored if n not in skip]

        delay = self._get_animation_delay("explore")
        if delay == 0:
            for node in nodes:
                self._paint_cell(node, C["explored"])
            self._anim_path(path, algo_id)
            return

        def flash(i=0):
            if i < len(nodes):
                self._paint_cell(nodes[i], C["explored"])
                self._anim_jobs.append(
                    self.root.after(delay, lambda: flash(i + 1)))
            else:
                self._anim_jobs.append(
                    self.root.after(100, lambda: self._anim_path(path, algo_id)))
        flash()

    def _anim_path(self, path, algo_id):
        skip = {self.env.start_coordinate, self.env.goal_coordinate}
        cells = [n for n in path if n not in skip]

        delay = self._get_animation_delay("path")
        if delay == 0:
            for cell in cells:
                self._paint_cell(cell, C["path"])
            self._stamp_icon(self.env.start_coordinate, "S", C["white"])
            self._stamp_icon(self.env.goal_coordinate,  "G", C["white"])
            if path:
                self._paint_robot(path[-1])
            self._finish(algo_id, bool(path))
            return

        def draw(i=0):
            if i < len(cells):
                self._paint_cell(cells[i], C["path"])
                self._anim_jobs.append(
                    self.root.after(delay, lambda: draw(i + 1)))
            else:
                self._stamp_icon(self.env.start_coordinate, "S", C["white"])
                self._stamp_icon(self.env.goal_coordinate,  "G", C["white"])
                if path:
                    self._anim_jobs.append(
                        self.root.after(180, lambda: self._anim_robot(path, algo_id)))
                else:
                    self._finish(algo_id, False)
        draw()

    def _anim_robot(self, path, algo_id):
        delay = self._get_animation_delay("robot")
        def move(i=0):
            self.display_canvas.delete("robot")
            if i < len(path):
                self._paint_robot(path[i])
                self._anim_jobs.append(
                    self.root.after(delay, lambda: move(i + 1)))
            else:
                self.display_canvas.delete("robot")
                self._finish(algo_id, True)
        move()

    def _finish(self, algo_id, ok):
        if ok:
            self._set_status(f"{algo_id} complete ✓", "#6EE7B7")
            if getattr(self, '_sequential_routing', False):
                # Update current position to the destination
                self.csp.remove_impassable_obstacle(*self.env.start_coordinate)
                self.env.reconfigure_start_state(*self.env.goal_coordinate)
                self.csp.remove_impassable_obstacle(*self.env.goal_coordinate)
                self.uncertainty.unregister_high_traffic_zone(*self.env.goal_coordinate)
                self._update_curr_loc_lbl()
                self._render_grid()
        else:
            self._set_status("No path found!", "#FCA5A5")
        self._running = False
        self._sequential_routing = False

    def _cancel_anims(self):
        for j in self._anim_jobs:
            self.root.after_cancel(j)
        self._anim_jobs.clear()
        self.display_canvas.delete("robot")

    def _clear(self):
        if self._running:
            self._cancel_anims()
            self._running = False
        self.csp.erase_all_constraints()
        self.uncertainty.clear_uncertainty_parameters()
        self.decision.reset_decision_matrix()
        for w in (self.sv_algo, self.sv_steps, self.sv_nodes, self.sv_time,
                  self.mm_score, self.mm_best):
            w.config(text="—")
        self.sv_cost.config(text="Total Cost: —", fg=C["txt"])
        self.mm_trace.config(text="Awaiting run…")
        self.route_trace.config(text="Route: —")
        self.b_congest.config(text="—")
        self.b_delay.config(text="0")
        self.csp_status.config(text="Eliminating invalid paths…", fg=C["m3"])
        self.xai_text.config(state=tk.NORMAL)
        self.xai_text.delete("1.0", tk.END)
        self.xai_text.insert(tk.END,
            "> SYSTEM_RESET: Tactical HUD reset.\n"
            "> [PROTOCOL]: Select search algorithm to stream reasoning.")
        self.xai_text.config(state=tk.DISABLED)
        self._set_status("Ready", C["secondary"])
        self._update_curr_loc_lbl()
        self._render_grid()

    def _generate_random_obstacles(self):
        """Generates random obstacles based on slider density, avoiding landmarks/start/goal."""
        if self._running: return
        density = self.obstacle_density_var.get()
        self.csp.structural_obstacles_set.clear()
        
        rows = self.env.total_rows
        cols = self.env.total_cols
        total_cells = rows * cols
        num_obstacles = int(total_cells * density)
        
        start = self.env.start_coordinate
        goal = self.env.goal_coordinate
        
        count = 0
        attempts = 0
        while count < num_obstacles and attempts < 1000:
            attempts += 1
            r = random.randint(0, rows - 1)
            c = random.randint(0, cols - 1)
            if (r, c) == start or (r, c) == goal or (r, c) in LANDMARKS:
                continue
            if (r, c) not in self.csp.structural_obstacles_set:
                self.csp.register_impassable_obstacle(r, c)
                self.uncertainty.unregister_high_traffic_zone(r, c)
                count += 1
                
        self._render_grid()
        self._set_status(f"Generated {count} obstacles", C["m1"])

    def _refresh_bayes(self):
        """Randomise Bayesian priors, synchronize sliders, and run inference."""
        p_storm = round(random.uniform(0.0, 1.0), 2)
        p_incident = round(random.uniform(0.0, 1.0), 2)
        
        self.uncertainty.probability_of_storm = p_storm
        self.uncertainty.probability_of_road_incident = p_incident
        
        if hasattr(self, 'storm_var'):
            self.storm_var.set(p_storm)
            self.storm_lbl.config(text=f"{int(p_storm*100)}%")
        if hasattr(self, 'incident_var'):
            self.incident_var.set(p_incident)
            self.incident_lbl.config(text=f"{int(p_incident*100)}%")
            
        self._update_bayes_inference()

    def _update_bayes_inference(self):
        """Runs Bayesian network inference and updates UI stats."""
        self.b_storm.config(text=f"{self.uncertainty.probability_of_storm:.0%}")
        self.b_incident.config(text=f"{self.uncertainty.probability_of_road_incident:.0%}")
        prob, _, _ = self.uncertainty.evaluate_congestion_probability()
        self.b_congest.config(text=f"{prob:.0%}")
        self.b_delay.config(text=str(len(self.uncertainty.high_traffic_risk_cells)))