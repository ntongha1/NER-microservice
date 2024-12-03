import pickle

class HeirarchyRule(object):
    def __init__(self, rule_dict: dict):
        self.rule_dict = rule_dict
    def __setstate__(self, state):
        self.rule_dict = state["rule_dict"]
    def __getstate__(self):
        return {"rule_dict":self.rule_dict}
    def apply(self, extrated_codes: list):
        ec_labels = {code_info["label"] for code_info in extrated_codes}
        codes2remove = set()
        for code in ec_labels:
            if code in self.rule_dict:
                lower_codes = self.rule_dict[code]
                for lc in lower_codes:
                    if lc in ec_labels:
                        codes2remove.add(lc)
        output = []
        for code_info in extrated_codes:
            if not(code_info["label"] in codes2remove):
                output.append(code_info)
        return output
class ComboCodeRule(object):
    def __init__(self, rule_dict):
        self.rule_dict = rule_dict
    def apply(self, extrated_codes):
        pass

def load_rule(path):
    with open(path, "rb") as output_file:
        return pickle.load(output_file)