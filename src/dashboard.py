import time
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import clear_output

from src.database_service import get_records


class RobotDashboard:

    def show_live_dashboard(self):

        records = get_records()

        if not records:
            print("No data available.")
            return

        columns = [
            "id",
            "trait",
            "axis_1",
            "axis_2",
            "axis_3",
            "axis_4",
            "axis_5",
            "axis_6",
            "axis_7",
            "axis_8",
            "year",
            "month",
            "day",
            "time"
        ]

        df = pd.DataFrame(records, columns=columns)

        df["Timestamp"] = pd.to_datetime(
            df["year"].astype(str) + "-" +
            df["month"].astype(str) + "-" +
            df["day"].astype(str) + " " +
            df["time"].astype(str)
        )

        latest_time = df["Timestamp"].max()

        recent_data = df[
            df["Timestamp"] >= latest_time - pd.Timedelta(seconds=90)
        ]

        clear_output(wait=True)

        plt.figure(figsize=(12, 6))

        for axis in range(1, 9):
            plt.plot(
                recent_data["Timestamp"],
                recent_data[f"axis_{axis}"],
                label=f"Axis #{axis}"
            )

        plt.xlabel("Time")
        plt.ylabel("Axis Reading")
        plt.title("Real-Time Robot Dashboard")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()