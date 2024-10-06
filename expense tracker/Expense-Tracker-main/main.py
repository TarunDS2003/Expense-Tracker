import json
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QPainter
from PySide6.QtWidgets import (
    QApplication, QHeaderView, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QMessageBox
)
from PySide6.QtCharts import QChartView, QPieSeries, QChart
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.items = 0
        self.budget_limit = None
        self.file_path = "table_data.json"  # JSON file path
        self.ml_model = self.train_ml_model()  # Initialize machine learning model

        # Left Widget
        self.table = QTableWidget()
        self.table.setColumnCount(4)  # Added column for necessary/not necessary prediction
        self.table.setHorizontalHeaderLabels(["Description", "Price", "Action", "Necessary"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Chart
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.Antialiasing)

        # Right Widget
        self.description = QLineEdit()
        self.price = QLineEdit()
        self.add = QPushButton("Add")
        self.clear = QPushButton("Clear")
        self.quit = QPushButton("Quit")
        self.plot = QPushButton("Plot")
        self.total_expense_button = QPushButton("Total Expense")
        self.budget_label = QLabel("Set Budget Limit:")
        self.budget_input = QLineEdit()
        self.set_budget_button = QPushButton("Set Budget")

        # Disabling 'Add' button
        self.add.setEnabled(False)

        self.right = QVBoxLayout()
        self.right.addWidget(QLabel("Description"))
        self.right.addWidget(self.description)
        self.right.addWidget(QLabel("Price"))
        self.right.addWidget(self.price)
        self.right.addWidget(self.add)
        self.right.addWidget(self.plot)
        self.right.addWidget(self.chart_view)
        self.right.addWidget(self.clear)
        self.right.addWidget(self.quit)
        self.right.addWidget(self.total_expense_button)
        self.right.addWidget(self.budget_label)
        self.right.addWidget(self.budget_input)
        self.right.addWidget(self.set_budget_button)

        # QWidget Layout
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.table)
        self.layout.addLayout(self.right)
        self.setLayout(self.layout)

        # Signals and Slots
        self.add.clicked.connect(self.add_element)
        self.quit.clicked.connect(self.quit_application)
        self.plot.clicked.connect(self.plot_data)
        self.clear.clicked.connect(self.clear_table)
        self.description.textChanged[str].connect(self.check_disable)
        self.price.textChanged[str].connect(self.check_disable)
        self.total_expense_button.clicked.connect(self.calculate_and_show_total_expense)
        self.set_budget_button.clicked.connect(self.set_budget_limit)

        # Fill example data
        self.load_data_from_file()  # Load saved data

    def train_ml_model(self):
        # Dummy training data for demonstration
        training_data = [
            ("Groceries", True),               # Necessary
            ("Restaurant", False),             # Not Necessary
            ("Transportation", True),          # Necessary
            ("Entertainment", False),          # Not Necessary
            ("Rent", True),                    # Necessary
            ("Utility Bill", True),            # Necessary
            ("Gym Membership", False),         # Not Necessary
            ("Medical Expenses", True),        # Necessary
            ("Vacation", False),               # Not Necessary
            ("Insurance Premium", True),       # Necessary
            ("Online Subscription", False),    # Not Necessary
            ("Education Fees", True),          # Necessary
            ("Pet Supplies", False),           # Not Necessary
            ("Clothing", True),                # Necessary
            ("Gifts", False),                  # Not Necessary
            ("Home Maintenance", True),        # Necessary
            ("Phone Bill", True),              # Necessary
            ("Coffee Shop", False),            # Not Necessary
            ("Electronics", False),            # Not Necessary
            ("Car Maintenance", True),         # Necessary
            ("Charity Donation", True),        # Necessary
            ("Hobbies", False),                # Not Necessary
            ("Haircut", True),                 # Necessary
            ("Books", True),                  # Necessary
            ("Subscription Service", False),   # Not Necessary
            ("Dental Checkup", True),          # Necessary
            ("Furniture", True),               # Necessary
            ("Movie Tickets", False),          # Not Necessary
            ("Airfare", False),                # Not Necessary
            ("Grocery Delivery", True),        # Necessary
            ("Jewelry", False),                # Not Necessary
            ("Public Transport", True),        # Necessary
            ("Concert Tickets", False),        # Not Necessary
            ("Home Decor", True),              # Necessary
            ("Computer Software", False),      # Not Necessary
            ("Medical Insurance", True),       # Necessary
            ("Alcohol", False),                # Not Necessary
            ("Fitness Classes", False),        # Not Necessary
            ("Kitchen Appliances", True),      # Necessary
            ("Magazine Subscription", False),  # Not Necessary
            ("Dining Out", False),             # Not Necessary
            ("Electricity Bill", True),        # Necessary
            ("Travel Bag", True),              # Necessary
            ("Art Supplies", False),           # Not Necessary
            ("Pet Grooming", False),           # Not Necessary
            ("Home Security", True),           # Necessary
            ("Streaming Service", False),      # Not Necessary
            ("Dining Table", True),            # Necessary
            ("Gift Cards", False),             # Not Necessary
            ("Childcare", True),               # Necessary
            ("Yoga Classes", False),           # Not Necessary
            ("DIY Tools", True),               # Necessary
            ("Music Lessons", False),          # Not Necessary
            ("Outdoor Gear", True),            # Necessary
            ("Bike Repair", True),             # Necessary
            ("Dry Cleaning", True),            # Necessary
            ("Photography Equipment", False),  # Not Necessary
            ("Home Renovation", True),         # Necessary
            ("Video Games", False),            # Not Necessary
            ("Spa Treatment", False),          # Not Necessary
            ("Newspaper Subscription", False), # Not Necessary
            ("Grocery Coupons", True),         # Necessary
            ("Collectibles", False),           # Not Necessary
            ("Personal Trainer", False),       # Not Necessary
            ("Online Courses", True),          # Necessary
            ("Camping Gear", True),            # Necessary
            ("Utility Deposit", True),         # Necessary
            ("Convenience Store", False),      # Not Necessary
            ("Car Insurance", True),           # Necessary
            ("Cooking Classes", False),        # Not Necessary
            ("Home Theater System", True),     # Necessary
            ("Mobile App Subscription", False), # Not Necessary
            ("game",False)
            # Add more examples as needed
        ]


        # Preparing features and labels
        X_train = [data[0] for data in training_data]
        y_train = [data[1] for data in training_data]

        # Machine learning pipeline with CountVectorizer and RandomForestClassifier
        model = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', RandomForestClassifier())
        ])

        # Training the model
        model.fit(X_train, y_train)

        # Dummy accuracy score for demonstration
        y_pred = model.predict(X_train)
        print(f"Model Accuracy: {accuracy_score(y_train, y_pred):.2f}")

        return model

    @Slot()
    def add_element(self):
        des = self.description.text()
        price = self.price.text()

        try:
            price_item = QTableWidgetItem(f"{float(price):.2f}")
            price_item.setTextAlignment(Qt.AlignRight)

            self.table.insertRow(self.items)
            description_item = QTableWidgetItem(des)
            delete_button = QPushButton("Delete")
            delete_button.clicked.connect(self.delete_row)  # Connect delete button to delete_row slot

            self.table.setItem(self.items, 0, description_item)
            self.table.setItem(self.items, 1, price_item)
            self.table.setCellWidget(self.items, 2, delete_button)  # Add delete button to the cell

            # Predict if the expense is necessary or not
            prediction = self.ml_model.predict([des])[0]
            prediction_text = "Necessary" if prediction else "Not Necessary"

            necessary_item = QTableWidgetItem(prediction_text)
            necessary_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(self.items, 3, necessary_item)

            self.description.setText("")
            self.price.setText("")

            self.items += 1

            # Save data to JSON file
            self.save_data_to_file()

            # Recalculate total expense and check budget limit
            self.calculate_and_show_total_expense()

        except ValueError:
            print("Invalid input for price. Make sure to enter a valid price!")

    @Slot()
    def delete_row(self):
        button = self.sender()  # Get the button that was clicked
        if button:
            index = self.table.indexAt(button.pos())  # Get the index of the button's position
            if index.isValid():
                row = index.row()
                self.table.removeRow(row)
                self.items -= 1

                # Save data to JSON file
                self.save_data_to_file()

                # Recalculate total expense and check budget limit
                self.calculate_and_show_total_expense()

    @Slot()
    def check_disable(self, x):
        if not self.description.text() or not self.price.text():
            self.add.setEnabled(False)
        else:
            self.add.setEnabled(True)

    @Slot()
    def plot_data(self):
        series = QPieSeries()
        for i in range(self.table.rowCount()):
            text = self.table.item(i, 0).text()
            number = float(self.table.item(i, 1).text())
            series.append(text, number)

        chart = QChart()
        chart.addSeries(series)
        chart.legend().setAlignment(Qt.AlignLeft)
        self.chart_view.setChart(chart)

    @Slot()
    def quit_application(self):
        # Save data to JSON file when quitting application
        self.save_data_to_file()
        QApplication.quit()

    def load_data_from_file(self):
        try:
            with open(self.file_path, "r") as file:
                data = json.load(file)
                self.load_table_from_data(data)
                self.load_budget_from_data(data)

        except FileNotFoundError:
            print(f"File {self.file_path} not found. Starting with default data.")

        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.file_path}. Starting with default data.")

    def load_table_from_data(self, data):
        if "table_data" in data:
            table_data = data["table_data"]
            for row_data in table_data:
                description_item = QTableWidgetItem(row_data["description"])
                
                # Handle loading price as float
                try:
                    price_value = float(row_data["price"])
                except ValueError:
                    price_value = 0.0  # Default to 0.0 if conversion fails
                
                price_item = QTableWidgetItem(f"{price_value:.2f}")
                price_item.setTextAlignment(Qt.AlignRight)

                delete_button = QPushButton("Delete")
                delete_button.clicked.connect(self.delete_row)  # Connect delete button to delete_row slot

                row_position = self.table.rowCount()
                self.table.insertRow(row_position)
                self.table.setItem(row_position, 0, description_item)
                self.table.setItem(row_position, 1, price_item)
                self.table.setCellWidget(row_position, 2, delete_button)  # Add delete button to the cell

                # Predict if the expense is necessary or not
                prediction = self.ml_model.predict([row_data["description"]])[0]
                prediction_text = "Necessary" if prediction else "Not Necessary"

                necessary_item = QTableWidgetItem(prediction_text)
                necessary_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_position, 3, necessary_item)

                self.items += 1

            # Calculate and show initial total expense
            self.calculate_and_show_total_expense()

    def load_budget_from_data(self, data):
        if "budget_limit" in data:
            self.budget_limit = data["budget_limit"]
            self.budget_input.setText(str(self.budget_limit))

    def save_data_to_file(self):
        table_data = []
        for row in range(self.table.rowCount()):
            description = self.table.item(row, 0).text()
            price = self.table.item(row, 1).text()
            table_data.append({"description": description, "price": price})

        data = {"table_data": table_data, "budget_limit": self.budget_limit}

        with open(self.file_path, "w") as file:
            json.dump(data, file, indent=4)

    @Slot()
    def clear_table(self):
        self.table.setRowCount(0)
        self.items = 0

        # Clear budget limit and UI
        self.budget_limit = None
        self.budget_input.clear()
        self.budget_input.setEnabled(True)
        self.set_budget_button.setEnabled(True)

        # Save data to JSON file after clearing
        self.save_data_to_file()

    @Slot()
    def calculate_and_show_total_expense(self):
        total_expense = 0.0
        for row in range(self.table.rowCount()):
            price_text = self.table.item(row, 1).text()
            try:
                price = float(price_text)
                total_expense += price
            except ValueError:
                pass  # Ignore rows with non-numeric price or handle error

        # Display total expense
        self.total_expense_button.setText(f"Total Expense: {total_expense:.2f}")

        # Check budget limit
        if self.budget_limit is not None and total_expense > self.budget_limit:
            self.show_budget_exceeded_message()

    @Slot()
    def set_budget_limit(self):
        budget_text = self.budget_input.text()
        try:
            self.budget_limit = float(budget_text)
            self.budget_input.setEnabled(False)
            self.set_budget_button.setEnabled(False)

            # Save data to JSON file after setting budget
            self.save_data_to_file()

            # Calculate and show total expense after setting budget
            self.calculate_and_show_total_expense()

        except ValueError:
            print("Invalid input for budget limit. Make sure to enter a valid number.")

    def show_budget_exceeded_message(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Budget Exceeded")
        msg_box.setText("The total expense has exceeded the budget limit.")
        msg_box.exec()


class MainWindow(QMainWindow):
    def __init__(self, widget):
        super().__init__()
        self.setWindowTitle("Expense Tracker")

        # Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        # Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.exit_app)

        self.file_menu.addAction(exit_action)
        self.setCentralWidget(widget)

    @Slot()
    def exit_app(self, checked):
        # Save data to JSON file when quitting application
        widget.save_data_to_file()
        QApplication.quit()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    widget = Widget()
    window = MainWindow(widget)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
