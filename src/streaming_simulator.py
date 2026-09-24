import time
import pandas as pd

from src.database_service import insert_record


class StreamingSimulator:

    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.current_index = 0

    def nextDataPoint(self):

        if self.current_index >= len(self.data):
            return None

        data_point = self.data.iloc[self.current_index]
        self.current_index += 1

        return data_point

    def send_to_database(self, data_point):

        insert_record(
            data_point["Trait"],
            data_point["Axis #1"],
            data_point["Axis #2"],
            data_point["Axis #3"],
            data_point["Axis #4"],
            data_point["Axis #5"],
            data_point["Axis #6"],
            data_point["Axis #7"],
            data_point["Axis #8"],
            data_point["Time"]
        )

    def start_stream(self, dashboard):

        print("Starting robot data stream...")

        while True:

            data_point = self.nextDataPoint()

            if data_point is None:
                break

            self.send_to_database(data_point)

            dashboard.show_live_dashboard()

            if self.current_index < len(self.data):
                time.sleep(2)

        print("Streaming stopped.")