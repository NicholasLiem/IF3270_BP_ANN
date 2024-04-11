import json

class JsonModelParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_json_file()
        self.parse_model_data()

    def load_json_file(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"The file {self.filepath} was not found")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the file {self.filepath}")
            return None

    def parse_model_data(self):
        if self.data:
            self.case = self.data.get('case', {})
            self.model = self.case.get('model', {})
            self.input_data = self.case.get('input', [])
            self.weights = self.case.get('weights', [])
            self.expect = self.data.get('expect', {})
            raw_layers = self.model.get('layers', [])

            self.input_size = self.model.get('input_size')
            self.layers = [{'number_of_neurons': layer.get('number_of_neurons'),
                        'activation_function': layer.get('activation_function')}
                       for layer in raw_layers]

            self.output = self.expect.get('output', [])
            self.max_sse = self.expect.get('max_sse')

    @staticmethod
    def save_json_file(data, filepath):
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        except IOError:
            print(f"Could not save data to {filepath}")
