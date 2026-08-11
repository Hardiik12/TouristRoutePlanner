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
    "bg":          "#F0F4F8",
    "card":        "#FFFFFF",
    "border":      "#CBD5E1",
    "header":      "#1E3A5F",
    "header_txt":  "#FFFFFF",

    # Grid cells
    "free":        "#FFFFFF",
    "start":       "#10B981",
    "goal":        "#EF4444",
    "obstacle":    "#374151",
    "traffic":     "#F59E0B",
    "explored":    "#BAE6FD",
    "path":        "#2563EB",
    "robot":       "#7C3AED",
    "landmark":    "#FDE68A",

    # Module accent strips
    "m1": "#3B82F6",   # blue   – Environment
    "m2": "#10B981",   # green  – Search
    "m3": "#F59E0B",   # amber  – CSP
    "m4": "#EF4444",   # red    – Decision
    "m5": "#8B5CF6",   # purple – Uncertainty
    "m6": "#0EA5E9",   # sky    – Pipeline

    "txt":    "#1E293B",
    "txt2":   "#64748B",
    "white":  "#FFFFFF",
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
    """Tiny labelled value widget placed in a grid."""
    f = tk.Frame(parent, bg=C["card"],
                 highlightbackground=accent, highlightthickness=1)
    f.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
    tk.Label(f, text=label, bg=C["card"], fg=C["txt2"],
             font=("Helvetica", 7)).pack(anchor="w", padx=6, pady=(4, 0))
    val = tk.Label(f, text="—", bg=C["card"], fg=accent,
                   font=("Helvetica", 12, "bold"))
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

        # Two-column body
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self._build_left(body)
        self._build_right(body)

        self._render_grid()
        self.display_canvas.bind("<Button-1>", self._lclick)
        self.display_canvas.bind("<Button-3>", self._rclick)

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C["header"])
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="🗺  AI Tourist Route Planner  |  Project 26",
                 bg=C["header"], fg=C["white"],
                 font=("Helvetica", 13, "bold"), pady=10, padx=14).pack(side=tk.LEFT)
        self.status_lbl = tk.Label(hdr, text="● Ready",
                                   bg=C["header"], fg="#6EE7B7",
                                   font=("Helvetica", 10, "bold"), padx=14)
        self.status_lbl.pack(side=tk.RIGHT)

    # ── Left column: grid + landmarks ─────────────────────────────────────────

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=C["bg"])
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), pady=6)

        # MODULE 1 header
        m1_hdr = tk.Frame(left, bg=C["m1"])
        m1_hdr.pack(fill=tk.X)
        tk.Label(m1_hdr, text="① Environment & State Space",
                 bg=C["m1"], fg=C["white"],
                 font=("Helvetica", 9, "bold"), pady=4, padx=8, anchor="w").pack(fill=tk.X)

        # ── Mode Selector toolbar ─────────────────────────────────────────
        mode_card = tk.Frame(left, bg=C["card"],
                             highlightbackground=C["border"], highlightthickness=1)
        mode_card.pack(fill=tk.X, pady=(0, 2))
        tk.Label(mode_card, text="✏  Click Mode",
                 bg="#E2E8F0", fg=C["txt"],
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

        # Draw active mode initially
        self._refresh_mode_buttons()

        # Current-mode indicator
        self.mode_indicator = tk.Label(mode_card,
            text="Mode: ⬛ Obstacle  |  Left-click on grid to paint",
            bg=C["card"], fg=C["txt2"], font=("Helvetica", 7), pady=3)
        self.mode_indicator.pack(fill=tk.X, padx=8)

        # ── Grid canvas ───────────────────────────────────────────────────
        canvas_card = tk.Frame(left, bg=C["card"],
                               highlightbackground=C["border"], highlightthickness=1)
        canvas_card.pack()
        grid_px = self.env.total_cols * CELL_PX
        self.display_canvas = tk.Canvas(canvas_card, width=grid_px, height=grid_px,
                                        bg=C["free"], highlightthickness=0)
        self.display_canvas.pack(padx=4, pady=4)

        # ── Landmark Route Selector Card ──────────────────────────────────────────────
        route_card = tk.Frame(left, bg=C["card"],
                              highlightbackground=C["border"], highlightthickness=1)
        route_card.pack(fill=tk.X, pady=(6, 0))
        tk.Label(route_card, text="📍 Tourist Landmark Routing",
                 bg=C["m1"], fg=C["white"],
                 font=("Helvetica", 8, "bold"), padx=8, pady=4, anchor="w").pack(fill=tk.X)

        rf = tk.Frame(route_card, bg=C["card"], padx=8, pady=6)
        rf.pack(fill=tk.X)

        # Current Location display
        lf = tk.Frame(rf, bg=C["card"])
        lf.pack(fill=tk.X, pady=(0, 4))
        self.curr_loc_lbl = tk.Label(lf, text="📍 Current: Beach 🏖 (0,0)", bg=C["card"], fg=C["m1"],
                                     font=("Helvetica", 8, "bold"), anchor="w")
        self.curr_loc_lbl.pack(fill=tk.X)

        # Destination Combobox
        df = tk.Frame(rf, bg=C["card"])
        df.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        tk.Label(df, text="🎯 Next Destination", bg=C["card"], fg=C["txt2"],
                 font=("Helvetica", 7, "bold")).pack(anchor="w", pady=(0, 2))
        self.goal_lm_var = tk.StringVar(value="Museum 🏛 (3,5)")
        self.goal_lm_cb = ttk.Combobox(df, textvariable=self.goal_lm_var, state="readonly", font=("Helvetica", 8), width=12)
        self.goal_lm_cb['values'] = ("Beach 🏖 (0,0)", "Museum 🏛 (3,5)", "Temple ⛩ (10,12)", "Park 🌳 (14,8)", "Mall 🛍 (19,19)")
        self.goal_lm_cb.pack(fill=tk.X)

        # Algorithm Combobox
        af = tk.Frame(rf, bg=C["card"])
        af.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        tk.Label(af, text="⚙ Algorithm", bg=C["card"], fg=C["txt2"],
                 font=("Helvetica", 7, "bold")).pack(anchor="w", pady=(0, 2))
        self.route_algo_var = tk.StringVar(value="A*")
        self.route_algo_cb = ttk.Combobox(af, textvariable=self.route_algo_var, state="readonly", font=("Helvetica", 8), width=8)
        self.route_algo_cb['values'] = ("A*", "BFS", "UCS", "DFS")
        self.route_algo_cb.pack(fill=tk.X)

        # Start Journey Button
        btn_f = tk.Frame(route_card, bg=C["card"], padx=8, pady=4)
        btn_f.pack(fill=tk.X)

        def _route_landmarks():
            if self._running: return
            # Parse coordinates
            lm_coords = {
                "Beach 🏖 (0,0)": (0, 0),
                "Museum 🏛 (3,5)": (3, 5),
                "Temple ⛩ (10,12)": (10, 12),
                "Park 🌳 (14,8)": (14, 8),
                "Mall 🛍 (19,19)": (19, 19)
            }
            start_coord = self.env.start_coordinate
            goal_coord = lm_coords[self.goal_lm_var.get()]

            if start_coord == goal_coord:
                self._set_status("Already at destination!", C["goal"])
                return

            # Set sequential routing flag
            self._sequential_routing = True

            # Update goal state (start is current position)
            self.csp.remove_impassable_obstacle(*goal_coord)
            self.env.reconfigure_goal_state(*goal_coord)
            self.csp.remove_impassable_obstacle(*goal_coord)
            self.uncertainty.unregister_high_traffic_zone(*goal_coord)

            self._render_grid()
            # Run the selected algorithm
            self._run(self.route_algo_var.get())

        btn = pill_btn(btn_f, "🚀 Start Journey", C["m2"],
                       cmd=_route_landmarks, w=388, h=32)
        btn.pack(pady=2)



        # ── Cell type legend ──────────────────────────────────────────────
        type_card = tk.Frame(left, bg=C["card"],
                             highlightbackground=C["border"], highlightthickness=1)
        type_card.pack(fill=tk.X, pady=(6, 0))
        tk.Label(type_card, text="Cell Colour Key",
                 bg="#E2E8F0", fg=C["txt"],
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


    # ── Right column: all control modules ─────────────────────────────────────

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=C["bg"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=6)

        self._build_module2(right)
        self._build_module3(right)

        mid = tk.Frame(right, bg=C["bg"])
        mid.pack(fill=tk.X)
        self._build_module4(mid)
        self._build_module5(mid)

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

        # Stats grid
        sg = tk.Frame(c, bg=C["card"])
        sg.pack(fill=tk.X, pady=(6, 0))
        sg.columnconfigure(0, weight=1)
        sg.columnconfigure(1, weight=1)
        sg.columnconfigure(2, weight=1)
        sg.columnconfigure(3, weight=1)

        self.sv_algo  = mini_stat(sg, "Algorithm",     C["m5"], 0, 0)
        self.sv_steps = mini_stat(sg, "Path Steps",    C["m1"], 1, 0)
        self.sv_nodes = mini_stat(sg, "Nodes Expanded",C["m3"], 2, 0)
        self.sv_time  = mini_stat(sg, "Compute Time",  C["m2"], 3, 0)

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
        c = section_card(parent, "⑤ Reasoning Under Uncertainty", C["m5"], pady=4)
        tk.Label(c, text="Bayesian Network", bg=C["card"], fg=C["txt"],
                 font=("Helvetica", 8, "bold")).pack(anchor="w")

        bf = tk.Frame(c, bg=C["card"])
        bf.pack(fill=tk.X)
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

        tk.Button(c, text="🔄 Refresh Priors", bg=C["m5"], fg=C["white"],
                  font=("Helvetica", 7, "bold"), relief="flat", cursor="hand2",
                  command=self._refresh_bayes).pack(pady=(6, 0))

    # MODULE 6 – Integrated Pipeline + XAI ────────────────────────────────────
    def _build_module6(self, parent):
        c = section_card(parent, "⑥ Integrated Pipeline & Explainable Output  (Reasoning Trace)", C["m6"], pady=4)

        # Route trace row
        self.route_trace = tk.Label(c, text="Route: —", bg=C["card"],
                                    fg=C["m6"], font=("Helvetica", 9, "bold"),
                                    wraplength=420, justify=tk.LEFT)
        self.route_trace.pack(anchor="w", pady=(0, 4))

        self.xai_text = tk.Text(c, width=58, height=5,
                                font=("Helvetica", 8),
                                bg="#F8FAFC", fg=C["txt"],
                                relief="flat", wrap=tk.WORD,
                                highlightbackground=C["border"],
                                highlightthickness=1)
        self.xai_text.pack(fill=tk.X)
        self.xai_text.insert(tk.END,
            "Run any algorithm above to see the AI reasoning trace here.\n"
            "The system will explain:\n"
            "  • Which path was chosen and why\n"
            "  • How CSP constraints filtered options\n"
            "  • Bayesian uncertainty impact on route cost")
        self.xai_text.config(state=tk.DISABLED)

    def _build_lab(self, parent):
        c = section_card(parent, "🧪 Algorithm Laboratory  (Performance Comparison)", C["m1"], pady=4)

        # Comparison triggers
        btn_f = tk.Frame(c, bg=C["card"])
        btn_f.pack(fill=tk.X, pady=(2, 6))

        pill_btn(btn_f, "🧪 Run Comparison Analysis", C["m1"],
                 cmd=self._run_lab_comparison, w=388, h=30).pack(pady=2)

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

    def _set_status(self, msg, color="#6EE7B7"):
        self.status_lbl.config(text=f"● {msg}", fg=color)

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
        self.xai_text.config(state=tk.NORMAL)
        self.xai_text.delete("1.0", tk.END)
        self.xai_text.insert(tk.END, xai)
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
            "Run any algorithm above to see the AI reasoning trace here.")
        self.xai_text.config(state=tk.DISABLED)
        self._set_status("Ready", "#6EE7B7")
        self._update_curr_loc_lbl()
        self._render_grid()

    def _refresh_bayes(self):
        """Randomise Bayesian priors and update display."""
        self.uncertainty.probability_of_storm        = round(random.uniform(0.1, 0.7), 2)
        self.uncertainty.probability_of_road_incident = round(random.uniform(0.05, 0.5), 2)
        self.b_storm.config(text=f"{self.uncertainty.probability_of_storm:.0%}")
        self.b_incident.config(text=f"{self.uncertainty.probability_of_road_incident:.0%}")