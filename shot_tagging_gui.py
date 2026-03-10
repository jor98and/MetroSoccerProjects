import csv
import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass
from typing import Optional

FIELDNAMES = [
    "shot_distance",
    "shot_angle",
    "GameId",
    "Id",
    "RelatedId",
    "EventId",
    "Half",
    "Time",
    "Player",
    "Action",
    "Type",
    "Foot",
    "Result",
    "Touches on Ball",
    "X",
    "Y",
    "Y standing",
    "pressure_level",
    "assist_type",
    "open_play_or_set_piece",
    "first_time",
    "rebound",
    "goalkeeper_state",
    "shot_phase",
    "competition_level",
    "notes",
]

HALF_PITCH_LENGTH = 60
PITCH_WIDTH = 80
GOAL_WIDTH = 8


@dataclass
class ShotEvent:
    shot_distance: str = ""
    shot_angle: str = ""
    GameId: str = ""
    Id: str = ""
    RelatedId: str = ""
    EventId: str = ""
    Half: str = "1"
    Time: str = "00:00"
    Player: str = ""
    Action: str = "Shot"
    Type: str = "Open Play"
    Foot: str = "Right"
    Result: str = "Saved"
    Touches_on_Ball: str = "1"
    X: str = ""
    Y: str = ""
    Y_standing: str = ""
    pressure_level: str = "Unknown"
    assist_type: str = "Unknown"
    open_play_or_set_piece: str = "Open Play"
    first_time: str = "No"
    rebound: str = "No"
    goalkeeper_state: str = "Normal"
    shot_phase: str = "Possession"
    competition_level: str = "High School"
    notes: str = ""

    def to_csv_row(self):
        return {
            "shot_distance": self.shot_distance,
            "shot_angle": self.shot_angle,
            "GameId": self.GameId,
            "Id": self.Id,
            "RelatedId": self.RelatedId,
            "EventId": self.EventId,
            "Half": self.Half,
            "Time": self.Time,
            "Player": self.Player,
            "Action": self.Action,
            "Type": self.Type,
            "Foot": self.Foot,
            "Result": self.Result,
            "Touches on Ball": self.Touches_on_Ball,
            "X": self.X,
            "Y": self.Y,
            "Y standing": self.Y_standing,
            "pressure_level": self.pressure_level,
            "assist_type": self.assist_type,
            "open_play_or_set_piece": self.open_play_or_set_piece,
            "first_time": self.first_time,
            "rebound": self.rebound,
            "goalkeeper_state": self.goalkeeper_state,
            "shot_phase": self.shot_phase,
            "competition_level": self.competition_level,
            "notes": self.notes,
        }


class PitchCanvas(ttk.Frame):
    def __init__(self, master, on_pick):
        super().__init__(master)
        self.on_pick = on_pick
        self.canvas_width = 560
        self.canvas_height = 420
        self.margin = 20
        self.canvas = tk.Canvas(
            self,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#2e7d32",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.marker = None
        self.vector_items = []
        self.draw_half_pitch()
        self.canvas.bind("<Button-1>", self._clicked)

    def draw_half_pitch(self):
        c = self.canvas
        w, h = self.canvas_width, self.canvas_height
        m = self.margin
        c.delete("all")

        c.create_rectangle(m, m, w - m, h - m, outline="white", width=2)
        c.create_line(m, m, w - m, m, fill="white", width=3)

        cx = w / 2
        pen_depth = self.yards_to_canvas_y(18)
        six_depth = self.yards_to_canvas_y(6)
        pen_half_width = self.yards_to_canvas_x(22)
        six_half_width = self.yards_to_canvas_x(10)

        c.create_rectangle(cx - pen_half_width, m, cx + pen_half_width, m + pen_depth, outline="white", width=2)
        c.create_rectangle(cx - six_half_width, m, cx + six_half_width, m + six_depth, outline="white", width=2)

        pen_spot_y = m + self.yards_to_canvas_y(12)
        c.create_oval(cx - 2, pen_spot_y - 2, cx + 2, pen_spot_y + 2, fill="white", outline="white")

        goal_half = self.yards_to_canvas_x(GOAL_WIDTH / 2)
        c.create_line(cx - goal_half, m - 6, cx + goal_half, m - 6, fill="white", width=4)
        c.create_line(cx - goal_half, m - 6, cx - goal_half, m, fill="white", width=2)
        c.create_line(cx + goal_half, m - 6, cx + goal_half, m, fill="white", width=2)

        c.create_text(w / 2, h - 8, text="Normalized attacking half: goal at top", fill="white", font=("Arial", 11, "bold"))

    def yards_to_canvas_x(self, yards):
        usable_w = self.canvas_width - 2 * self.margin
        return yards / (PITCH_WIDTH / 2) * (usable_w / 2)

    def yards_to_canvas_y(self, yards):
        usable_h = self.canvas_height - 2 * self.margin
        return yards / HALF_PITCH_LENGTH * usable_h

    def _clicked(self, event):
        pitch_x, pitch_y = self.canvas_to_pitch(event.x, event.y)
        cx, cy = self.pitch_to_canvas(pitch_x, pitch_y)
        self.place_marker(cx, cy)
        self.draw_vectors(cx, cy)
        self.on_pick(pitch_x, pitch_y)

    def place_marker(self, x, y):
        if self.marker is not None:
            self.canvas.delete(self.marker)
        r = 6
        self.marker = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="red", outline="white")

    def draw_vectors(self, x, y):
        for item in self.vector_items:
            self.canvas.delete(item)
        self.vector_items = []

        m = self.margin
        cx = self.canvas_width / 2
        goal_half = self.yards_to_canvas_x(GOAL_WIDTH / 2)
        left_post = (cx - goal_half, m)
        right_post = (cx + goal_half, m)
        goal_center = (cx, m)

        self.vector_items.append(self.canvas.create_line(x, y, goal_center[0], goal_center[1], fill="yellow", width=2))
        self.vector_items.append(self.canvas.create_line(x, y, left_post[0], left_post[1], fill="white", dash=(4, 3)))
        self.vector_items.append(self.canvas.create_line(x, y, right_post[0], right_post[1], fill="white", dash=(4, 3)))

    def clear_marker(self):
        if self.marker is not None:
            self.canvas.delete(self.marker)
            self.marker = None
        for item in self.vector_items:
            self.canvas.delete(item)
        self.vector_items = []

    def canvas_to_pitch(self, x, y):
        m = self.margin
        usable_w = self.canvas_width - 2 * m
        usable_h = self.canvas_height - 2 * m
        x = min(max(x, m), self.canvas_width - m)
        y = min(max(y, m), self.canvas_height - m)

        pitch_x = ((x - m) / usable_w) * PITCH_WIDTH - (PITCH_WIDTH / 2)
        pitch_y = ((y - m) / usable_h) * HALF_PITCH_LENGTH
        return round(pitch_x, 2), round(pitch_y, 2)

    def pitch_to_canvas(self, pitch_x, pitch_y):
        m = self.margin
        usable_w = self.canvas_width - 2 * m
        usable_h = self.canvas_height - 2 * m
        x = m + ((pitch_x + (PITCH_WIDTH / 2)) / PITCH_WIDTH) * usable_w
        y = m + (pitch_y / HALF_PITCH_LENGTH) * usable_h
        return x, y


class ShotLoggerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Minimal Soccer Shot Logger")
        self.geometry("1220x840")
        self.output_path = os.path.abspath("tagged_shots.csv")
        self.current_pitch_x: Optional[float] = None
        self.current_pitch_y: Optional[float] = None

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(3, weight=1)
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(left, text="Shot Location (Attacking Half)", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.pitch = PitchCanvas(left, self.set_pitch_location)
        self.pitch.grid(row=1, column=0, sticky="n")

        self.location_var = tk.StringVar(value="Shot location: not selected")
        self.geometry_var = tk.StringVar(value="Distance: --   Angle: --")
        ttk.Label(left, textvariable=self.location_var, font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="nw", pady=(10, 0))
        ttk.Label(left, textvariable=self.geometry_var, font=("Arial", 10)).grid(row=3, column=0, sticky="nw", pady=(4, 0))

        ttk.Label(right, text="Shot Details", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.form = ttk.Frame(right)
        self.form.grid(row=1, column=0, sticky="nsew")
        self.form.columnconfigure(1, weight=1)

        self.vars = {}
        self._build_form()
        self._build_bottom(right)
        self._bind_shortcuts()

        self.ensure_csv_exists()
        self.clear_form()
        self.refresh_table()

    def _build_form(self):
        fields = [
            ("shot_distance", ttk.Entry),
            ("shot_angle", ttk.Entry),
            ("GameId", ttk.Entry),
            ("Id", ttk.Entry),
            ("Half", ttk.Combobox, ["1", "2", "ET1", "ET2"]),
            ("Time", ttk.Entry),
            ("Player", ttk.Entry),
            ("Type", ttk.Combobox, ["Open Play", "Header", "Volley", "Free Kick", "Penalty", "Corner"]),
            ("Foot", ttk.Combobox, ["Right", "Left", "Head", "Other"]),
            ("Result", ttk.Combobox, ["Goal", "Saved", "Off Target", "Blocked", "Post"]),
            ("Touches on Ball", ttk.Combobox, ["1", "2", "3", "4+"]),
            ("pressure_level", ttk.Combobox, ["None", "Light", "Moderate", "Heavy", "Unknown"]),
            ("assist_type", ttk.Combobox, ["Pass", "Through Ball", "Cross", "Cutback", "Dribble", "Rebound", "Set Piece", "Unknown"]),
            ("open_play_or_set_piece", ttk.Combobox, ["Open Play", "Set Piece"]),
            ("first_time", ttk.Combobox, ["Yes", "No"]),
            ("rebound", ttk.Combobox, ["Yes", "No"]),
            ("goalkeeper_state", ttk.Combobox, ["Normal", "1v1", "Off Line", "Unsighted", "Unknown"]),
            ("shot_phase", ttk.Combobox, ["Possession", "Transition", "Counter", "Set Piece", "Scramble"]),
            ("competition_level", ttk.Combobox, ["High School", "Club", "College", "USL2", "Other"]),
            ("notes", ttk.Entry),
        ]

        for row, spec in enumerate(fields):
            label = spec[0]
            widget_cls = spec[1]
            ttk.Label(self.form, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
            var = tk.StringVar()
            self.vars[label] = var

            if widget_cls is ttk.Combobox:
                values = spec[2]
                widget = ttk.Combobox(self.form, textvariable=var, values=values, state="readonly")
                if values:
                    var.set(values[0])
            else:
                widget = widget_cls(self.form, textvariable=var)
                if label in {"shot_distance", "shot_angle"}:
                    widget.configure(state="readonly")

            widget.grid(row=row, column=1, sticky="ew", padx=4, pady=4)

    def _build_bottom(self, parent):
        controls = ttk.Frame(parent)
        controls.grid(row=2, column=0, sticky="ew", pady=(12, 6))

        ttk.Button(controls, text="Save Shot", command=self.save_shot).pack(side="left", padx=4)
        ttk.Button(controls, text="Clear", command=self.clear_form).pack(side="left", padx=4)
        ttk.Button(controls, text="Choose CSV", command=self.choose_csv).pack(side="left", padx=4)

        self.csv_path_var = tk.StringVar(value=f"Output: {self.output_path}")
        ttk.Label(parent, textvariable=self.csv_path_var).grid(row=3, column=0, sticky="w")

        columns = ["Id", "Time", "Player", "Result", "X", "Y"]
        self.table = ttk.Treeview(parent, columns=columns, show="headings", height=10)
        for col in columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=90 if col in {"Id", "Time", "Result", "X", "Y"} else 150)
        self.table.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        parent.rowconfigure(4, weight=1)

    def _bind_shortcuts(self):
        self.bind("<Return>", lambda e: self.save_shot())
        self.bind("g", lambda e: self.vars["Result"].set("Goal"))
        self.bind("s", lambda e: self.vars["Result"].set("Saved"))
        self.bind("b", lambda e: self.vars["Result"].set("Blocked"))
        self.bind("o", lambda e: self.vars["Result"].set("Off Target"))
        self.bind("p", lambda e: self.vars["Type"].set("Penalty"))
        self.bind("h", lambda e: self.vars["Foot"].set("Head"))

    def compute_distance_and_angle(self, x, y):
        distance = (x ** 2 + y ** 2) ** 0.5
        left_post_x = -GOAL_WIDTH / 2
        right_post_x = GOAL_WIDTH / 2
        left_vec = (left_post_x - x, -y)
        right_vec = (right_post_x - x, -y)
        dot = left_vec[0] * right_vec[0] + left_vec[1] * right_vec[1]
        mag1 = (left_vec[0] ** 2 + left_vec[1] ** 2) ** 0.5
        mag2 = (right_vec[0] ** 2 + right_vec[1] ** 2) ** 0.5
        if mag1 == 0 or mag2 == 0:
            angle_deg = 0.0
        else:
            cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            angle_deg = math.degrees(math.acos(cos_theta))
        return round(distance, 2), round(angle_deg, 2)

    def set_pitch_location(self, x, y):
        self.current_pitch_x = x
        self.current_pitch_y = y
        distance, angle = self.compute_distance_and_angle(x, y)
        self.location_var.set(f"Shot location: lateral={x}, distance_from_goal={y}")
        self.geometry_var.set(f"Distance: {distance}   Angle: {angle}°")
        self.vars["shot_distance"].set(str(distance))
        self.vars["shot_angle"].set(str(angle))

    def choose_csv(self):
        path = filedialog.asksaveasfilename(
            title="Choose output CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=os.path.basename(self.output_path),
        )
        if not path:
            return
        self.output_path = path
        self.csv_path_var.set(f"Output: {self.output_path}")
        self.ensure_csv_exists()
        self.refresh_table()
        self.clear_form()

    def ensure_csv_exists(self):
        if not os.path.exists(self.output_path):
            with open(self.output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()

    def next_auto_id(self) -> str:
        try:
            with open(self.output_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            numeric_ids = [int(r["Id"]) for r in rows if str(r.get("Id", "")).isdigit()]
            return str(max(numeric_ids) + 1) if numeric_ids else "1"
        except Exception:
            return "1"

    def validate(self) -> bool:
        if self.current_pitch_x is None or self.current_pitch_y is None:
            messagebox.showerror("Missing location", "Click the half-pitch to set shot location.")
            return False
        if not self.vars["Time"].get().strip():
            messagebox.showerror("Missing time", "Enter the video timestamp manually.")
            return False
        return True

    def build_event(self) -> ShotEvent:
        return ShotEvent(
            shot_distance=self.vars["shot_distance"].get().strip(),
            shot_angle=self.vars["shot_angle"].get().strip(),
            GameId=self.vars["GameId"].get().strip(),
            Id=self.vars["Id"].get().strip() or self.next_auto_id(),
            RelatedId="",
            EventId="",
            Half=self.vars["Half"].get().strip(),
            Time=self.vars["Time"].get().strip(),
            Player=self.vars["Player"].get().strip(),
            Type=self.vars["Type"].get().strip(),
            Foot=self.vars["Foot"].get().strip(),
            Result=self.vars["Result"].get().strip(),
            Touches_on_Ball=self.vars["Touches on Ball"].get().strip(),
            X=str(self.current_pitch_x),
            Y=str(self.current_pitch_y),
            Y_standing=str(self.current_pitch_y),
            pressure_level=self.vars["pressure_level"].get().strip(),
            assist_type=self.vars["assist_type"].get().strip(),
            open_play_or_set_piece=self.vars["open_play_or_set_piece"].get().strip(),
            first_time=self.vars["first_time"].get().strip(),
            rebound=self.vars["rebound"].get().strip(),
            goalkeeper_state=self.vars["goalkeeper_state"].get().strip(),
            shot_phase=self.vars["shot_phase"].get().strip(),
            competition_level=self.vars["competition_level"].get().strip(),
            notes=self.vars["notes"].get().strip(),
        )

    def save_shot(self):
        if not self.validate():
            return
        event = self.build_event()
        with open(self.output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(event.to_csv_row())
        self.refresh_table()
        self.clear_form(keep_game=True)

    def clear_form(self, keep_game=False):
        game_id = self.vars.get("GameId", tk.StringVar()).get() if keep_game else ""
        for _, var in self.vars.items():
            var.set("")

        self.vars["GameId"].set(game_id)
        self.vars["Id"].set(self.next_auto_id())
        self.vars["Half"].set("1")
        self.vars["Type"].set("Open Play")
        self.vars["Foot"].set("Right")
        self.vars["Result"].set("Saved")
        self.vars["Touches on Ball"].set("1")
        self.vars["pressure_level"].set("Unknown")
        self.vars["assist_type"].set("Pass")
        self.vars["open_play_or_set_piece"].set("Open Play")
        self.vars["first_time"].set("No")
        self.vars["rebound"].set("No")
        self.vars["goalkeeper_state"].set("Normal")
        self.vars["shot_phase"].set("Possession")
        self.vars["competition_level"].set("High School")
        self.vars["shot_distance"].set("")
        self.vars["shot_angle"].set("")

        self.current_pitch_x = None
        self.current_pitch_y = None
        self.location_var.set("Shot location: not selected")
        self.geometry_var.set("Distance: --   Angle: --")
        self.pitch.clear_marker()

    def refresh_table(self):
        for item in self.table.get_children():
            self.table.delete(item)
        if not os.path.exists(self.output_path):
            return
        with open(self.output_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))[-12:]
        for row in rows:
            self.table.insert(
                "",
                "end",
                values=(row.get("Id"), row.get("Time"), row.get("Player"), row.get("Result"), row.get("X"), row.get("Y")),
            )


if __name__ == "__main__":
    app = ShotLoggerApp()
    app.mainloop()
