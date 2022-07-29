from app.common.base import PostprocessorBase

number_to_character = {
    "0": "D",
    "2": "Z",
    "4": "L",
    "8": "B",
    "7": "Z"
}

character_to_number = {
    "D": "0",
    "Z": "2",
    "L": "4",
    "B": "8",
    "O": "0",
    "C": "0",
    "S": "9",
    "F": "6",
    "E": "8"
}


class FormatPostprocessor(PostprocessorBase):
    # def _postprocess(self, results):
    #     valid_results = [result for result in results if len(result) in [9, 10]]
    #     valid_results = list(set(valid_results))
    #     return valid_results
    def _postprocess(self, result):
        org_plate_str = result.value
        filtered_plate_str = org_plate_str.replace('-', '').replace('.', '')
        if len(filtered_plate_str) == 8:
            result.value = self._process_lp_string(filtered_plate_str, 2)
        elif len(filtered_plate_str) == 9:
            result.value = self._process_lp_string(filtered_plate_str, 2, 3)
        else:
            result.value = ''
        return result

    def _process_lp_string(self, lp_string, *argv):
        new_str = ''
        for i, lp_char in enumerate(lp_string):
            if i in argv:
                if lp_char in number_to_character.keys():
                    new_str += number_to_character.get(lp_char)
                    continue
                else:
                    new_str += lp_char
                    continue
            if lp_char in character_to_number.keys():
                new_str += character_to_number.get(lp_char)
            else:
                new_str += lp_char
        return new_str
