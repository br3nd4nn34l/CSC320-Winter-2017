import os, sys, shutil

# Constants / basic functions

# Building paths out of folder lists
def make_path(folder_lst):
    return (os.sep).join(folder_lst)

# Run location
home = "home"
cdf = "cdf"
submission_folder = make_path(["Submission", "A2"])
instruction_folder = "Instructions"

# Helper for copying entire directories
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

# Copy source to target, will replace target if it exists
def copy(source, target):
    if os.path.isdir(source):
        copytree(source, target)
    elif os.path.isfile(source):
        shutil.copyfile(source, target)

# Determines whether or not this is running on my desktop or cdf computers
def get_location():
    if sys.platform == "win32":
        return home
    else:
        return cdf

# Determines which path to draw files from
def get_location_based_path(location):
    home_project_location = make_path(["G:", "Computer Science Work"])
    cdf_project_location = make_path([os.path.expanduser("~")])
    location_dict = {home: home_project_location,
                     cdf: cdf_project_location}
    cur_CS_work_path = location_dict[location]
    A2_folder = make_path([cur_CS_work_path, "CSC320-Winter-2017", "Assignments", "Assignment 2"])
    return A2_folder

# Makes the submission folder path
def submission_path(assignment_folder):
    return make_path([assignment_folder, submission_folder])


if __name__ == '__main__':

    # Figure out where we are
    location = get_location()

    # Figure out source and destination paths
    A2_folder = get_location_based_path(location)
    submission_folder = submission_path(A2_folder)

    # Clear the submission folder for writing
    shutil.rmtree(submission_folder)

    # Remake the submission folder and it's subdirectories
    os.mkdir(make_path([submission_folder]))
    for folder in ["code", "report", "extras"]:
        os.mkdir(make_path([submission_folder, folder]))


    # Mapping of sources to destinations:
    source_to_dest = {
                      # The checklist
                      make_path([A2_folder, instruction_folder, "CHECKLIST.txt"]):
                      make_path([submission_folder, "CHECKLIST.txt"]),

                      # Code folder
                      make_path([A2_folder, "Code"]):
                      make_path([submission_folder, "code"]),

                      # Report folder
                      make_path([A2_folder, "Report"]):
                      make_path([submission_folder, "report"]),

                      # Extras
                      make_path([A2_folder, "Extras"]):
                      make_path([submission_folder, "extras"])}

    # Copy source stuff to destinations
    for src_dst_pair in source_to_dest.items():
        print "Copying files from {src} to {dst}".format(src=src_dst_pair[0],
                                                         dst=src_dst_pair[1])
        copy(*src_dst_pair)

