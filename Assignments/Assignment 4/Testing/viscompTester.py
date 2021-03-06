import os, cv2 as cv, numpy as np, re, pprint
import sys

pp = pprint.PrettyPrinter(indent=4)

def make_path(folder_lst):
    return (os.sep).join(folder_lst)


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


def run_command(command_str):
    """
    Uses the operating system to run command_str
    """
    print "\n\n" + command_str
    os.system(command_str)

# Returns whether or not this is running on my home PC
def running_at_home():
    return sys.platform == "win32"


# Defining constants
home_project_location = make_path(["G:", "Computer Science Work"])
cdf_project_location = make_path([os.path.expanduser("~")])
location_dict = {True: home_project_location,
                 False: cdf_project_location}

old_bkg_folder = "Back"
comp_folder = "Comp"
new_bkg_folder = "NewBack"
result_folder = "Results"
my_program_result = "Mine"
ref_program_result = "Ref"


def safe_lookup(dct, key):
    if key not in dct:
        return None
    else:
        return dct[key]


def make_cmd_string(py_path, # Path to python
                    vc_path, # Path to executable
                    src, trg, out, # Image paths
                    k=False, iters=False, patch_size=False, alpha=False, w=False, # Numerical constants
                    init_NNF=False, nlm=False, nlm_h=False,
                    dis_rand=False, dis_prop=False, # Iteration params
                    nnf_img=False, part_res=False, nnf_vecs=False, rec_src=False, # NNF variables
                    nnf_ss=False, nnf_lw=False, nnf_lc=False,  # More NNF variables
                    MPL_serv=False, tmpdir=False):

    exec_path = """{py_name} \"{viscomp_path}\" """.format(py_name=py_path, viscomp_path=vc_path)

    return exec_path + create_cmd_args(src, trg, out, # Image paths
                    k, iters, patch_size, alpha, w, # Numerical constants
                    init_NNF, nlm, nlm_h,
                    dis_rand, dis_prop, # Iteration params
                    nnf_img, part_res, nnf_vecs, rec_src, # NNF variables
                    nnf_ss, nnf_lw, nnf_lc,  # More NNF variables
                    MPL_serv, tmpdir)

# If a variable has nothing associated with it (e.g. disable-random)
# or a default value, mark as False (to indicate the tag
# shouldn't show up in the command)
def create_cmd_args(src, trg, out, # Image paths
                    k, iters, patch_size, alpha, w, # Numerical constants
                    init_NNF, nlm, nlm_h,
                    dis_rand, dis_prop, # Iteration params
                    nnf_img, part_res, nnf_vecs, rec_src, # NNF variables
                    nnf_ss, nnf_lw, nnf_lc,  # More NNF variables
                    MPL_serv, tmpdir): # Some bullshit (???)

    # Dictionary of actual tag names to input names:
    tag_name_dct = {"source":src, "target":trg, "output":out,
                    "k":k, "iters":iters, "patch-size":patch_size, "alpha":alpha, "w":w,
                    "init-nnf":init_NNF, "nlm":nlm, "nlm_h":nlm_h,
                    "disable-random":dis_rand, "disable-propagation":dis_prop,
                    "nnf-image":nnf_img, "partial-results":part_res, "nnf-vectors":nnf_vecs, "rec-source":rec_src,
                    "nnf-subsampling":nnf_ss, "nnf-line-width":nnf_lw, "nnf-line-color":nnf_lc,
                    "server":MPL_serv, "tmpdir":tmpdir}

    # Format the list of command arguments
    arg_lst = filter(lambda x: x is not None,
                     [create_associated_command(*tup) for tup in tag_name_dct.items()])

    # Join the list of cmd args together
    return " ".join(arg_lst)


# Returns the None if there is no valid associated value
# Otherwise creates a valid command value pairing
def create_associated_command(tag, value):
    if not value:
        return None
    elif value is True:
        ret_str = "--{tag}"
    elif type(value) is str:
        ret_str = "--{tag} \"{val}\""
    elif type(value) in [int, float]:
        ret_str = "--{tag} {val}"
    return ret_str.format(tag=tag,
                          val=value)

# Maps Location to Python Executable
exec_map = {True:"""C:\ProgramData\Anaconda3\envs\python27\python.exe""",
            False:"""/local/bin/X11/python"""}

# Main program
if __name__ == '__main__':

    # Figuring out the location of the A3 Folder
    at_home = running_at_home()
    cur_CS_work_path = location_dict[at_home]
    A4_folder = make_path([cur_CS_work_path, "CSC320-Winter-2017",
                           "Assignments", "Assignment 4"])
    viscompPy_path = make_path([A4_folder, "Code", "viscomp.py"])
    testImgs_path = make_path([A4_folder, "test_images"])
    python_path = exec_map[at_home]

    # Dictionary of test images to their folders
    imgs_to_paths = {
        "Canyon": make_path([testImgs_path, "canyon"]),
        "Deer": make_path([testImgs_path, "deer"]),
        "Jag": make_path([testImgs_path, "jaguar"]),
        "Jag2" : make_path([testImgs_path, "jaguar2"]),
        "Jag3": make_path([testImgs_path, "jaguar3"]),
        "Raptor": make_path([testImgs_path, "raptor"]),
        "Stormtrooper": make_path([testImgs_path, "stormtrooper"]),
        "Doge": make_path([testImgs_path, "doge"]),
        "Mine":make_path([testImgs_path, "mine"])
    }

    # Path to output image folder
    output_path = make_path([A4_folder, "Output Images"])


    # Dictionary of test names to their command strings
    test_dct = {

        "Jag2 All": make_cmd_string(
            python_path,
            viscompPy_path,
            make_path([imgs_to_paths["Jag2"], "source.png"]),
            make_path([imgs_to_paths["Jag2"], "target.png"]),
            make_path([output_path, "Jag2 All", "Jag2 All"]),
            k=3,
            dis_prop=False, dis_rand=False,
            nnf_img=True, nnf_vecs=True,
            rec_src=True),

        "Jag Denoising Control 0.5wk3ps7h25": make_cmd_string(
            python_path,
            viscompPy_path,
            make_path([imgs_to_paths["Jag"], "source_noise.png"]),
            make_path([imgs_to_paths["Jag"], "source_noise.png"]),
            make_path([output_path, "Jag Denoising Control 0.5wk3ps7h25", "Jag Denoising Control 0.5wk3ps7h25"]),
            nlm=True, nlm_h=25, patch_size=7,
            k=3, iters=3, w=315,
            nnf_img=True, nnf_vecs=True,
            rec_src=True),

        "Jag Denoising k6": make_cmd_string(
            python_path,
            viscompPy_path,
            make_path([imgs_to_paths["Jag"], "source_noise.png"]),
            make_path([imgs_to_paths["Jag"], "source_noise.png"]),
            make_path([output_path, "Jag Denoising k6", "Jag Denoising k6"]),
            nlm=True, nlm_h=25, patch_size=7,
            k=6, iters=3, w=315,
            nnf_img=True, nnf_vecs=True,
            rec_src=True),

        "Jag Denoising ps9h31": make_cmd_string(
            python_path,
            viscompPy_path,
            make_path([imgs_to_paths["Jag"], "source_noise.png"]),
            make_path([imgs_to_paths["Jag"], "source_noise.png"]),
            make_path([output_path, "Jag Denoising ps9h31", "Jag Denoising ps9h31"]),
            nlm=True, nlm_h=31, patch_size=9,
            k=3, iters=3, w=315,
            nnf_img=True, nnf_vecs=True,
            rec_src=True),

        "Jag Denoising w": make_cmd_string(
            python_path,
            viscompPy_path,
            make_path([imgs_to_paths["Jag"], "source_noise.png"]),
            make_path([imgs_to_paths["Jag"], "source_noise.png"]),
            make_path([output_path, "Jag Denoising w", "Jag Denoising w"]),
            nlm=True, nlm_h=25, patch_size=7,
            k=3, iters=3,
            nnf_img=True, nnf_vecs=True,
            rec_src=True),

        "My Denoising Control 0.5wk3ps7h25": make_cmd_string(
            python_path,
            viscompPy_path,
            make_path([imgs_to_paths["Mine"], "source_noise.png"]),
            make_path([imgs_to_paths["Mine"], "source_noise.png"]),
            make_path([output_path, "My Denoising Control 0.5wk3ps7h25", "My Denoising Control 0.5wk3ps7h25"]),
            nlm=True, nlm_h=25, patch_size=7,
            k=3, iters=3, w=200,
            nnf_img=True, nnf_vecs=True,
            rec_src=True),

        "My Denoising k6": make_cmd_string(
            python_path,
            viscompPy_path,
            make_path([imgs_to_paths["Mine"], "source_noise.png"]),
            make_path([imgs_to_paths["Mine"], "source_noise.png"]),
            make_path([output_path, "My Denoising k6", "My Denoising k6"]),
            nlm=True, nlm_h=25, patch_size=7,
            k=6, iters=3, w=200,
            nnf_img=True, nnf_vecs=True,
            rec_src=True),

        "My Denoising ps9h31": make_cmd_string(
            python_path,
            viscompPy_path,
            make_path([imgs_to_paths["Mine"], "source_noise.png"]),
            make_path([imgs_to_paths["Mine"], "source_noise.png"]),
            make_path([output_path, "My Denoising ps9h31", "My Denoising ps9h31"]),
            nlm=True, nlm_h=31, patch_size=9,
            k=3, iters=3, w=200,
            nnf_img=True, nnf_vecs=True,
            rec_src=True),

        "My Denoising w": make_cmd_string(
            python_path,
            viscompPy_path,
            make_path([imgs_to_paths["Mine"], "source_noise.png"]),
            make_path([imgs_to_paths["Mine"], "source_noise.png"]),
            make_path([output_path, "My Denoising w", "My Denoising w"]),
            nlm=True, nlm_h=25, patch_size=7,
            k=3, iters=3,
            nnf_img=True, nnf_vecs=True,
            rec_src=True),
    }

    # List of tests that are okay to run with the algorithm
    # (because some actually can overload RAM or take forever)
    okay_to_run = ["Jag2 All"]

    # Clear out the folders before putting stuff inside them
    # for folder in os.listdir(output_path):
    #     for file in os.listdir(make_path([output_path, folder])):
    #         os.remove(make_path([output_path, folder, file]))

    if okay_to_run != [True]:
        for test in okay_to_run:
            run_command(test_dct[test])
    else:
        for test in test_dct:
            run_command(test_dct[test])





