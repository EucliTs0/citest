ClassTypes = dict[str, str | list[float]]


CLASSES: dict[str, ClassTypes] = {
    "0": {"name": "DENT", "color": [128, 128, 0], "color_name": "OLIVE_GREEN"},
    "1": {"name": "SCRATCH", "color": [255, 0, 0], "color_name": "RED"},
    "2": {"name": "CRACK", "color": [0, 255, 0], "color_name": "GREEN"},
    "3": {"name": "GLASS_SHATTER", "color": [0, 0, 255], "color_name": "BLUE"},
    "4": {"name": "LAMP_BROKEN", "color": [255, 255, 0], "color_name": "YELLOW"},
    "5": {"name": "TIRE_FLAT", "color": [255, 0, 255], "color_name": "MAGENTA"},
}