
import tkinter as tk
from gui import RoutePlannerCoreUI


def main():
    # Instantiate the primary window workspace framework
    application_window_root = tk.Tk()

    # Initialize the core graphical control dashboard
    RoutePlannerCoreUI(application_window_root)

    # Launch the execution loop
    application_window_root.mainloop()


if __name__ == "__main__":
    main()