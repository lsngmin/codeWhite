import os
def get_p(current_path=None) :
    path_lc = "/started/"
    path_sv = "/home/started/ultralytics/started1/"
    if current_path is None:
        path_g = path_lc if str(os.getcwd())[1] == 'U' else path_sv
        return path_g
    elif current_path == "ti":
        return "/home/started/ultralytics/datasets/train/images"
    elif current_path == "tl":
        return "/home/started/ultralytics/datasets/train/labels"
    elif current_path == "datasets":
        return "/home/started/ultralytics/datasets/"
    else:
        path_g = path_lc if str(os.getcwd())[1] == 'U' else path_sv
        finally_path = path_g + current_path
        return finally_path

print(get_p("trainModel/tld_detection/weights/best.pt"))