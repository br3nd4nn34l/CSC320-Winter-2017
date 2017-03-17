import os, cv2 as cv, numpy as np, re, pprint
import DistanceCheck as distcheck
import sys

pp = pprint.PrettyPrinter(indent=4)


def make_path(folder_lst):
    return (os.sep).join(folder_lst)


def extract_img_name(img_filename):
    img_name_pattern = "(?P<fileName>(.+))\.(?P<extension>(.+))"
    img_name_pattern = "(.+)\\" + os.sep + img_name_pattern
    img_name_regex = re.compile(img_name_pattern)
    match = re.match(img_name_regex, img_filename)
    if match != None:
        return match.group("fileName")
    else:
        return ""


def listdir_fullpath(target_folder):
    return [make_path([target_folder, f]) for f in os.listdir(target_folder)]


def compare_images(img1_ptr, img2_ptr):
    """(str, str) -> bool
    Returns true iff the images at img_ptr1 and img_ptr2 have the same
    dimensions and pixel values	everywhere.
    """
    if img1_ptr.shape == img2_ptr.shape:
        return (img1_ptr == img2_ptr).all()
    return False


def marked_diff_img(img1_ptr, img2_ptr):
    ret_img = img1_ptr
    ret_img[img1_ptr != img2_ptr] = np.array([0, 0, 255])
    return ret_img


def run_command(command_str):
    """
    Uses the operating system to run command_str
    """
    print "\n\n" + command_str
    os.system(command_str)


def difference_distribution(img1_ptr, img2_ptr):
    img1_ptr = img1_ptr.astype(np.int)
    img2_ptr = img2_ptr.astype(np.int)
    diff = (img1_ptr - img2_ptr).flatten()
    (num, freq) = np.unique(diff, return_counts=True)
    freq_dct = {}
    for i in range(len(num)):
        freq_dct[num[i]] = freq[i]
    return freq_dct


def make_matting_cmd(executable, cmd_dct):
    return matting_command_str.format(executable=exec_dct[executable],
                                      backAPath=cmd_dct["backAPath"],
                                      backBPath=cmd_dct["backBPath"],
                                      compAPath=cmd_dct["compAPath"],
                                      compBPath=cmd_dct["compBPath"],
                                      alphaOutPath=cmd_dct["alphaOutPath"],
                                      colOutPath=cmd_dct["colOutPath"])

def run_matting_cmd_dct(executable, cmd_dct):
    cmd_str = make_matting_cmd(executable, cmd_dct)
    run_command(cmd_str)

def make_composite_cmd(executable, cmd_dct):
    return composite_command_str.format(executable=exec_dct[executable],
                                        alphaInPath=cmd_dct["alphaInPath"],
                                        colInPath=cmd_dct["colInPath"],
                                        backInPath=cmd_dct["backInPath"],
                                        compOutPath=cmd_dct["compOutPath"])

def run_composite_cmds(executable, matting_cmd_dct):
    dct_lst = compositing_arg_lst(matting_cmd_dct)
    for dct in dct_lst:
        cmd_str = make_composite_cmd(executable, dct)
        run_command(cmd_str)

def run_matting_and_comping(cmd_dict, test_img_path, cur_location):
    my_matting_arg_dct_lst = list_for_matting(test_img_path, "Mine")
    ref_matting_arg_dct_lst = list_for_matting(test_img_path, "Ref")

    for i in range(len(my_matting_arg_dct_lst)):
        my_matting_cmd_dct = my_matting_arg_dct_lst[i]
        run_matting_cmd_dct("Mine", my_matting_cmd_dct)
        run_composite_cmds("Mine", my_matting_cmd_dct)

        ref_matting_cmd_dct = ref_matting_arg_dct_lst[i]
        if cur_location == "cdf":
            run_matting_cmd_dct("Ref", ref_matting_cmd_dct)
            run_composite_cmds("Ref", ref_matting_cmd_dct)

        test_result = True
        # test_result = test_matting(ref_matting_cmd_dct, my_matting_cmd_dct)

        if not test_result:
            print "Test failed:" \
                  "\n\tAlphas:" \
                  "\n\t\t{my_alpha}" \
                  "\n\t\t{ref_alpha}" \
                  "\n\tcolOut:" \
                  "\n\t\t{my_colOut}" \
                  "\n\t\t{ref_colOut}".format(my_alpha=my_matting_cmd_dct["alphaOutPath"],
                                              my_colOut=my_matting_cmd_dct["colOutPath"],
                                              ref_alpha=ref_matting_cmd_dct["alphaOutPath"],
                                              ref_colOut=ref_matting_cmd_dct["colOutPath"])

            # Figure out the difference distribution for alphas
            my_alpha = cv.imread(my_matting_cmd_dct["alphaOutPath"])
            ref_alpha = cv.imread(ref_matting_cmd_dct["alphaOutPath"])
            alpha_freq_dct = difference_distribution(my_alpha, ref_alpha)
            # print "Alpha difference distribution:"
            # pprint.pprint(alpha_freq_dct)

            # Give info on Manhattan norm
            print "Alpha Accuracy"
            distcheck.distances(my_alpha, ref_alpha)

            # Figure out the difference distribution for colout
            ref_colOut = cv.imread(ref_matting_cmd_dct["colOutPath"])
            my_colOut = cv.imread(my_matting_cmd_dct["colOutPath"])
            colOut_freq_dct = difference_distribution(my_colOut, ref_colOut)
            # print "colOut difference distribution:"
            # pprint.pprint(colOut_freq_dct)

            # Give info on Manhattan norm
            print "ColOut Accuracy:"
            distcheck.distances(my_colOut, ref_colOut)


def run_compositing(which_exec, alphaInPath, colInPath, backInPath,
                    compOutPath):
    composite_cmd_str = composite_command_str.format(executable=exec_dct[which_exec],
                                                     alphaInPath=alphaInPath,
                                                     colInPath=colInPath,
                                                     backInPath=backInPath,
                                                     compOutPath=compOutPath)
    run_command(composite_cmd_str)


def test_matting(ref_param_dct, mine_param_dct):
    # Open up the alpha for my solution
    my_soln_alpha_path = mine_param_dct["alphaOutPath"]
    my_soln_alpha_ptr = cv.imread(my_soln_alpha_path)

    # Open up the alpha for ref solution
    ref_soln_alpha_path = ref_param_dct["alphaOutPath"]
    ref_soln_alpha_ptr = cv.imread(ref_soln_alpha_path)

    # Compare the alpha solutions
    alpha_same = compare_images(my_soln_alpha_ptr, ref_soln_alpha_ptr)

    # Open up the colOut for my solution
    my_soln_colOut_path = mine_param_dct["colOutPath"]
    my_soln_colOut_ptr = cv.imread(my_soln_colOut_path)

    # Open up the colOut for ref solution
    ref_soln_colOut_path = ref_param_dct["colOutPath"]
    ref_soln_colOut_ptr = cv.imread(ref_soln_colOut_path)

    # Compare the colOut solutions
    colOut_same = compare_images(my_soln_colOut_ptr, ref_soln_colOut_ptr)

    # Return whether or not my solution produced the same result as the reference solution
    return (alpha_same and colOut_same)


def list_for_matting(test_img_path, which_solution):
    ret_lst = []

    for size_folder in listdir_fullpath(test_img_path):

        for subject_folder in listdir_fullpath(size_folder):

            result_path = make_path([subject_folder, result_folder])
            if not os.path.exists(result_path):
                os.mkdir(result_path)

            my_result_path = make_path([result_path, my_program_result])
            if not os.path.exists(my_result_path):
                os.mkdir(my_result_path)

            ref_result_path = make_path([result_path, ref_program_result])
            if not os.path.exists(ref_result_path):
                os.mkdir(ref_result_path)

            targ_res_folder_dct = {"Mine": my_result_path,
                                   "Ref": ref_result_path}

            A_str = "A.jpg"
            B_str = "B.jpg"

            backgroundFolderPath = make_path([subject_folder, old_bkg_folder])
            backAPath = make_path([backgroundFolderPath, A_str])
            backBPath = make_path([backgroundFolderPath, B_str])

            foregroundFolderPath = make_path([subject_folder, comp_folder])
            compAPath = make_path([foregroundFolderPath, A_str])
            compBPath = make_path([foregroundFolderPath, B_str])

            # Clear the target results folder (for my solution, don't delete the precious CDF solutions!
            if which_solution == "Mine":
                for file in os.listdir(targ_res_folder_dct[which_solution]):
                    os.remove(make_path([targ_res_folder_dct[which_solution], file]))

            alphaOutPath = make_path([targ_res_folder_dct[which_solution],
                                      "(compA, compB) = ({compA}, {compB}) Alpha.tif"])

            colOutPath = make_path([targ_res_folder_dct[which_solution],
                                    "colOut from {compA}.tif"])

            # Let (compA, compB) = (A, B)
            ret_lst += [{"backAPath": backAPath,
                         "backBPath": backBPath,
                         "compAPath": compAPath,
                         "compBPath": compBPath,
                         "alphaOutPath": alphaOutPath.format(
                             compA=extract_img_name(compAPath),
                             compB=extract_img_name(compBPath)),
                         "colOutPath": colOutPath.format(
                             compA=extract_img_name(compAPath)),
                         "resultsFolder": targ_res_folder_dct[which_solution],
                         "subjectFolder": subject_folder}]

            # Now do (compA, compB) = (B, A)
            # ret_lst += [{"backAPath": backBPath,
            #              "backBPath": backAPath,
            #              "compAPath": compBPath,
            #              "compBPath": compAPath,
            #              "alphaOutPath": alphaOutPath.format(
            #                  compA=extract_img_name(compBPath),
            #                  compB=extract_img_name(compAPath)),
            #              "colOutPath": colOutPath.format(
            #                  compA=extract_img_name(compBPath)),
            #              "resultsFolder": targ_res_folder_dct[which_solution],
            #              "subjectFolder": subject_folder}]

    return ret_lst

def compositing_arg_lst(matting_arg_dct):
    ret_lst = []

    backgrounds_folder = make_path([matting_arg_dct["subjectFolder"], new_bkg_folder])

    for bg in listdir_fullpath(backgrounds_folder):
        bg_name = extract_img_name(bg)
        new_comp_name = "(compA, compB) = ({compA}, {compB}) on {new_bg}.tif".format(
            compA=extract_img_name(matting_arg_dct["compAPath"]),
            compB=extract_img_name(matting_arg_dct["compBPath"]),
            new_bg=bg_name)

        ret_lst += [{"alphaInPath": matting_arg_dct["alphaOutPath"],
                     "colInPath": matting_arg_dct["colOutPath"],
                     "backInPath": bg,
                     "compOutPath": make_path([matting_arg_dct["resultsFolder"], new_comp_name])}]

    return ret_lst



# Defining constants
home_project_location = make_path(["G:", "Computer Science Work"])
cdf_project_location = make_path([os.path.expanduser("~")])
location_dict = {"home": home_project_location,
                 "cdf": cdf_project_location}

old_bkg_folder = "Back"
comp_folder = "Comp"
new_bkg_folder = "NewBack"
result_folder = "Results"
my_program_result = "Mine"
ref_program_result = "Ref"

# For running the matting process
matting_command_str = """{executable} --matting \
                --backA "{backAPath}" \
                --backB "{backBPath}" \
                --compA "{compAPath}" \
                --compB "{compBPath}" \
                --alphaOut "{alphaOutPath}" \
                --colOut "{colOutPath}\""""

# For running the composite process
composite_command_str = """{executable} --compositing \
	            --alphaIn "{alphaInPath}" \
	 			--colIn "{colInPath}" \
	            --backIn "{backInPath}" \
	            --compOut "{compOutPath}\""""

if __name__ == '__main__':

    if sys.platform == "win32":
        cur_location = "home"
    else:
        cur_location = "cdf"

    cur_CS_work_path = location_dict[cur_location]
    A1_folder = make_path([cur_CS_work_path, "CSC320-Winter-2017", "Assignments", "Assignment 1"])
    viscompPy_path = make_path([A1_folder, "partA", "viscomp.py"])

    # Strings for specifying which program to run
    exec_dct = {"Ref": "\"{contents}\"".format(contents=make_path([A1_folder, "partA",
                                                                   "viscomp.cdf", "viscomp"])),
                "Mine": """python "{vc_path}\"""".format(vc_path=viscompPy_path)}

    # Path to the test images
    test_img_path = make_path([A1_folder, "TestImages"])

    # Run matting, test for correctness
    run_matting_and_comping(exec_dct, test_img_path, cur_location)
