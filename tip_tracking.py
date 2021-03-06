import numpy as np
import os, glob, imageio, sys, time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def save_image(data, filename):
    sizes = np.shape(data)
    fig = plt.figure(figsize=(1,1))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap = plt.get_cmap("bone"))
    plt.savefig(filename,dpi = 1200)
    plt.close()

def calc_velocity(box_dat1, box_dat2, edge_file=None, frame_num1=None, frame_num2=None):
    dist = distance_matrix(box_dat1, box_dat2) # calculate euclidean distances between all high-confidence tips
    min_dists = np.amin(dist, axis=0) # Find the minimum distance between boxes
    arg_min_dist = np.argmin(dist, axis=0) # Find the index of the minimum distance
    mid_y = (box_dat2[np.arange(len(min_dists)), 0] + box_dat1[arg_min_dist, 0])/2 # y coord of segment midpoint
    mid_x = (box_dat2[np.arange(len(min_dists)), 1] + box_dat1[arg_min_dist, 1])/2 # x coord of segment midpoint
    delta_y = box_dat2[np.arange(len(min_dists)), 0] - box_dat1[arg_min_dist, 0] # Change in y
    delta_x = box_dat2[np.arange(len(min_dists)), 1] - box_dat1[arg_min_dist, 1] # Change in x
    norm_dy, norm_dx = delta_y/min_dists, delta_x/min_dists # normalize by distance
    if edge_file != None:
        rev_dist = distance_matrix(box_dat2, box_dat1)
        rev_min_dists = np.amin(dist, axis=0) # Find the minimum distance between boxes
        rev_arg_min_dist = np.argmin(dist, axis=0) # Find the index of the minimum distance
        buffer = ""
        for i in range(len(rev_min_dists)):
            buffer += F"{frame_num2}_{i:02d},{frame_num1}_{rev_arg_min_dist[i]:02d},{box_dat2[i,0]},{box_dat2[i,1]},{box_dat1[rev_arg_min_dist[i],0]},{box_dat1[rev_arg_min_dist[i],1]},{rev_min_dists[i]}\n"
        with open(edge_file, "a+") as ofile:
            ofile.write(buffer)
    return min_dists, norm_dy, norm_dx, mid_y, mid_x

def time_scatter_plot(times, intervals, pref):
    plt.scatter(times, intervals)
    plt.xlabel("Frame")
    plt.ylabel("Speed of tip movement (um/min)")
    plt.title("Hyphal tip speed progression")
    plt.savefig(F"{pref}speed_vs_time.jpg", dpi=1000)

def interval_hist(intervals, pref):
    plt.hist(intervals, bins = 30)
    plt.xlabel("Speed of tip movement (um/min)")
    plt.ylabel("Count")
    plt.title("Distribution of hyphal tip speeds")
    plt.savefig(F"{pref}speed_distribution.jpg", dpi=1000)

def segment_image(split, im):
    h, w = im.shape[1:3]
    h_pix, w_pix = int(h/split), int(w/split)
    im_list = []
    for x, y in zip(np.repeat(np.arange(split), split), np.tile(np.arange(split), split)):
        im_list.append(im[:,h_pix*x:h_pix*(x+1),w_pix*y:w_pix*(y+1),:])
    return im_list

def use_model(PREF, PATH_TO_CKPT='./training/frozen_inference_graph_v4.pb',
    PATH_TO_LABELS='./annotations/label_map.pbtxt', PATH_TO_IMS = './test_ims/',
    PATH_TO_ANNOT_IMS='./model_annots/', CSV_ONLY=False, FRAME_HEIGHT=989.9,
    FRAME_WIDTH=1319.9, FRAME_TIME=1.0, CONF_THR=0.3, OUTLIER_PROP=0.80,
    NUM_CLASSES=1, PATH_TO_CSV=None, SPEED_DAT_CSV=None, LOG_FILE=None,
    EDGELIST_FILE=None, REANNOTATE=True, SPLIT=None,
    FILE_EXT="jpg"):

    '''
    Args:
        PREF: Image file prefix.
        PATH_TO_CKPT: Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_LABELS: List of the strings that is used to add correct label for each box.
        PATH_TO_IMS: Path to image files.
        PATH_TO_ANNOT_IMS: Path to directory to store annotated images.
        CSV_ONLY: True if only comma-seperated value file outputs is desired.
        FRAME_HEIGHT: Frame HEIGHT in um, depends on microscope and magnification.
        FRAME_WIDTH: Frame width in um, depends on microscope and magnification.
        FRAME_TIME: Minutes between frames.
        CONF_THR: Confidence threshold to use for annotations, as a float.
        OUTLIER_PROP: Proportion of distances above which are considered outliers.
        NUM_CLASSES: Number of classes to detect
        PATH_TO_CSV: Path to exported CSV of box annotations.
        SPEED_DAT_CSV: Name for speed data file.
        LOG_FILE: Name for log file for timing data.
        EDGELIST_FILE: Name for a file to export nearest
        REANNOTATE: If rerunning annotation should be performed.
        SPLIT: Int (None) - If not none, segments image into SPLIT^2 images before prediction.
        FILE_EXT: str ('jpg') File extension of images used for annotation. Must be readable by matplotlib.plyplot.imread().
    '''
    if LOG_FILE != None:
        d_graph, l_and_c, box_time, int_create_time, int_exp_time, vid_exp_time = 0., 0., 0., 0., 0., 0.
        start = time.clock()
    CONF_PER = int(100 * CONF_THR)
    if PATH_TO_CSV == None:
        PATH_TO_CSV = F"box_data_{PREF}.csv"
    if SPEED_DAT_CSV == None:
        SPEED_DAT_CSV = F"{PREF}speed_data.csv"
    if EDGELIST_FILE != None:
        with open(EDGELIST_FILE, "w") as ofile:
            ofile.write("node_1,node_2,y1,x1,y2,x2,dist\n")

    # Load a (frozen) Tensorflow model into memory.
    if REANNOTATE:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        if LOG_FILE != None:
            d_graph = time.clock()
        # Loading label map
        # Label maps map indices to category names
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        if LOG_FILE != None:
            l_and_c = time.clock()
        if not os.path.exists(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh/") and not CSV_ONLY:
            os.mkdir(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh/")
        elif not CSV_ONLY:
                print("Overwritting annotated images")
        if not os.path.exists(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh_w_hist/") and not CSV_ONLY:
            os.mkdir(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh_w_hist/")
        nn_clock_start = time.clock()
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                with open(PATH_TO_CSV, "w") as file:
                    file.write("Frame,box1,box2,box3,box4,score,class\n")
                    test_ims = glob.glob(F"{PATH_TO_IMS}{PREF}*.{FILE_EXT}")
                    test_ims.sort()
                    all_ims = []
                    for i in test_ims:
                        # Read frame from camera
                        image_np = plt.imread(i).copy()
                        if len(image_np.shape) == 2 or image_np.shape[2] == 1:
                            image_np = np.stack((image_np,)*3, axis=-1)
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        if SPLIT == None:
                            # Extract image tensor
                            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                            # Extract detection boxes
                            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                            # Extract detection scores
                            scores = detection_graph.get_tensor_by_name('detection_scores:0')
                            # Extract detection classes
                            classes = detection_graph.get_tensor_by_name('detection_classes:0')
                            # Extract number of detectionsd
                            num_detections = detection_graph.get_tensor_by_name(
                                'num_detections:0')
                            (boxes, scores, classes, num_detections) = sess.run(
                                [boxes, scores, classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded}) # Actual detection.
                            # Visualization of the results of a detection.
                            buffer = ""
                            for x in range(np.squeeze(boxes).shape[0]):
                                buffer += F"{os.path.basename(i).split('.')[0]},{np.squeeze(boxes)[x,0]},{np.squeeze(boxes)[x,1]},{np.squeeze(boxes)[x,2]},{np.squeeze(boxes)[x,3]},{np.squeeze(scores)[x]},{np.squeeze(classes)[x]}\n"
                            file.write(buffer)
                            if not CSV_ONLY:
                                vis_util.visualize_boxes_and_labels_on_image_array(
                                    image_np,
                                    np.squeeze(boxes),
                                    np.squeeze(classes).astype(np.int32),
                                    np.squeeze(scores),
                                    category_index,
                                    use_normalized_coordinates=True,
                                    line_thickness=4,
                                    max_boxes_to_draw=100,
                                    min_score_thresh=CONF_THR)
                                save_image(image_np, F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh/{os.path.basename(i).split('.')[0]}_annot.jpg")
                                all_ims.append(image_np)
                            print(i, "\t", time.clock() - nn_clock_start)
                        else:
                            # Extract image tensor
                            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                            spl_fac = 1/SPLIT
                            im_segmented_list = segment_image(SPLIT, image_np_expanded)
                            boxes_all, scores_all, classes_all = [np.empty((1,0,4)), np.empty((1,0)), np.empty((1,0))]
                            for r, c, im_seg in zip(np.repeat(np.arange(SPLIT), SPLIT), np.tile(np.arange(SPLIT), SPLIT), im_segmented_list):
                                # Actual detection.
                                boxes_seg = detection_graph.get_tensor_by_name('detection_boxes:0')
                                # Extract detection scores
                                scores_seg = detection_graph.get_tensor_by_name('detection_scores:0')
                                # Extract detection classes
                                classes_seg = detection_graph.get_tensor_by_name('detection_classes:0')
                                # Extract number of detectionsd
                                num_detections_seg = detection_graph.get_tensor_by_name(
                                    'num_detections:0')
                                buffer = ""
                                (boxes_seg, scores_seg, classes_seg, num_detections_seg) = sess.run(
                                    [boxes_seg, scores_seg, classes_seg, num_detections_seg],
                                    feed_dict={image_tensor: im_seg})
                                # Visualization of the results of a detection.
                                buffer = ""
                                for x in range(np.squeeze(boxes_seg).shape[0]):
                                    buffer += F"{os.path.basename(i).split('.')[0]},{np.squeeze(boxes_seg)[x,0]*spl_fac + r*spl_fac},{np.squeeze(boxes_seg)[x,1]*spl_fac + c*spl_fac},"
                                    buffer += F"{np.squeeze(boxes_seg)[x,2]*spl_fac + r*spl_fac},{np.squeeze(boxes_seg)[x,3]*spl_fac + c*spl_fac},{np.squeeze(scores_seg)[x]},{np.squeeze(classes_seg)[x]}\n"
                                file.write(buffer)
                                boxes_seg = np.stack((boxes_seg[:,:,0]*spl_fac + r*spl_fac,
                                                            boxes_seg[:,:,1]*spl_fac + c*spl_fac,
                                                            boxes_seg[:,:,2]*spl_fac + r*spl_fac,
                                                            boxes_seg[:,:,3]*spl_fac + c*spl_fac), axis=-1)
                                boxes_all = np.hstack((boxes_all, boxes_seg))
                                scores_all = np.hstack((scores_all, scores_seg))
                                classes_all = np.hstack((classes_all, classes_seg))
                            if not CSV_ONLY:
                                print("Boxes shape, ", boxes_all.shape, classes_all.shape, scores_all.shape)
                                vis_util.visualize_boxes_and_labels_on_image_array(
                                    image_np,
                                    np.squeeze(boxes_all),
                                    np.squeeze(classes_all).astype(np.int32),
                                    np.squeeze(scores_all),
                                    category_index,
                                    use_normalized_coordinates=True,
                                    line_thickness=4,
                                    max_boxes_to_draw=100*SPLIT**2,
                                    min_score_thresh=CONF_THR)
                                save_image(image_np, F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh/{os.path.basename(i).split('.')[0]}_annot.jpg")
                                all_ims.append(image_np)
                            print(i, "\t", time.clock() - nn_clock_start)
        if LOG_FILE != None:
            box_time = time.clock()
        # Saves annotated frames to a .mp4 file
        if not CSV_ONLY:
            imageio.mimsave(F"./model_annots/{PREF}annot_{CONF_PER}pc_thresh.mp4", all_ims, fps=15)                # Display output

    ims = glob.glob(F"{PATH_TO_IMS}{PREF}*.{FILE_EXT}") # Gets list of all saved images
    ims.sort() # Sorts alphabetically
    ims_base = [] # List to store basenames without extension
    for i in ims:
        ims_base.append(os.path.basename(i).split(".")[0])

    intervals = np.array([]) # array to store distance intervals between tips
    times = np.array([]) # array to store times for each interval
    mid_y_arr, mid_x_arr = np.array([]), np.array([]) # x and y coords midpoint of tip-connecting segment
    dy_comps, dx_comps = np.array([]), np.array([]) # normalized x and y components of tip movement
    medians = [] # List to store median distance for each frame
    box_dat = pd.read_csv(PATH_TO_CSV) # Box position data, also tip position data

    # Extract center coordinates of boxes
    box_dat["ycoord"] = (box_dat.box1 + box_dat.box3)/2*(FRAME_HEIGHT)
    box_dat["xcoord"] = (box_dat.box2 + box_dat.box4)/2*(FRAME_WIDTH)

    # Select frame which meet these criterea: 1) the first frame and 2) are above confidence threshold
    box_dat_sub1 = np.array(box_dat[(box_dat.Frame == ims_base[0]) & (box_dat.score > CONF_THR)].iloc[:,7:])

    if CSV_ONLY:
        for i in range(len(ims_base)-1): # For each frame
            box_dat_sub2 = np.array(box_dat[(box_dat.Frame == ims_base[i+1]) & (box_dat.score > CONF_THR)].iloc[:,7:]) # Extract the next frame's box data
            if (box_dat_sub1.shape[0] > 0) & (box_dat_sub2.shape[0] > 0):
                if EDGELIST_FILE != None:
                    min_dists, norm_dy, norm_dx, mid_y, mid_x = calc_velocity(box_dat_sub1, box_dat_sub2, edge_file=EDGELIST_FILE, frame_num1=ims_base[i], frame_num2=ims_base[i+1])
                else:
                    min_dists, norm_dy, norm_dx, mid_y, mid_x = calc_velocity(box_dat_sub1, box_dat_sub2)
                intervals = np.concatenate((intervals, min_dists)) # Add minimum distances to intervals array
                mid_y_arr = np.concatenate((mid_y_arr, mid_y))
                mid_x_arr = np.concatenate((mid_x_arr, mid_x))
                dy_comps = np.concatenate((dy_comps, norm_dy))
                dx_comps = np.concatenate((dx_comps, norm_dx))
                times = np.concatenate((times, np.repeat(i*FRAME_TIME, len(min_dists)))) # Add frame number to times
            box_dat_sub1 = box_dat_sub2
        if LOG_FILE != None:
            int_create_time = time.clock()
        # Create dataframe to store output
        speed_dat = pd.DataFrame({"Time" : times, "Speed" : intervals, "Y_component" : dy_comps, "X_component" : dx_comps, "Y_mid" : mid_y_arr, "X_mid" : mid_x_arr})
        # Export dataframe as CSV file
        speed_dat.to_csv(SPEED_DAT_CSV)
        if LOG_FILE != None:
            int_exp_time = time.clock()

    else:
        for i in range(len(ims_base)-1): # For each frame
            im2 = plt.imread(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh/{ims_base[i+1]}_annot.jpg") # Read in the next frame as an array
            box_dat_sub2 = np.array(box_dat[(box_dat.Frame == ims_base[i+1]) & (box_dat.score > CONF_THR)].iloc[:,7:]) # Extract the next frame's box data
            if (box_dat_sub1.shape[0] > 0) & (box_dat_sub2.shape[0] > 0):
                if EDGELIST_FILE != None:
                    min_dists, norm_dy, norm_dx, mid_y, mid_x = calc_velocity(box_dat_sub1, box_dat_sub2, edge_file=EDGELIST_FILE, frame_num1=ims_base[i], frame_num2=ims_base[i+1])
                else:
                    min_dists, norm_dy, norm_dx, mid_y, mid_x = calc_velocity(box_dat_sub1, box_dat_sub2)
                intervals = np.concatenate((intervals, min_dists)) # Add minimum distances to intervals array
                mid_y_arr = np.concatenate((mid_y_arr, mid_y))
                mid_x_arr = np.concatenate((mid_x_arr, mid_x))
                dy_comps = np.concatenate((dy_comps, norm_dy))
                dx_comps = np.concatenate((dx_comps, norm_dx))

                times = np.concatenate((times, np.repeat(i*FRAME_TIME, len(min_dists)))) # Add frame number to times
                if LOG_FILE != None:
                    int_create_time = time.clock()
                ints_wo_outliers = intervals[intervals < np.quantile(intervals, OUTLIER_PROP)] # Remove top proportion as outliers
            medians.append(np.median(ints_wo_outliers)) # Store median of distances

            plt.clf() # Clear figure
            # Set up figure with subplots
            fig = plt.figure()
            gs = fig.add_gridspec(3,2)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            ax3 = fig.add_subplot(gs[1:3,:])
            # Plot median line chart
            ax1.plot(medians)
            ax1.set_ylabel("Median tip speed (um/min)")
            ax1.set_xlabel("Frame number")
            ax1.set_title("Median hyphal tip speed")
            # Plot histogram of tip speeds/intervals
            ax2.hist(ints_wo_outliers, bins = 30)
            ax2.set_xlabel("Tip speed (um/min)")
            ax2.set_ylabel("Count")
            ax2.set_title("Distribution of hyphal tip speeds")
            ax2.axis("tight")
            # Show annotated images
            ax3.imshow(im2)
            ax3.axis("off")
            plt.tight_layout()
            # Save the annotated figure
            plt.savefig(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh_w_hist/{ims_base[i+1]}_annot_w_hist.jpg", dpi=400)
            # Next frame
            box_dat_sub1 = box_dat_sub2
        # Create dataframe to store output
        speed_dat = pd.DataFrame({"Time" : times, "Speed" : intervals, "Y_component" : dy_comps, "X_component" : dx_comps, "Y_mid" : mid_y_arr, "X_mid" : mid_x_arr})
        # Export dataframe as CSV file
        speed_dat.to_csv(SPEED_DAT_CSV)
        # Slice the intervals and time array to remove "outliers"
        ints_wo_outliers = intervals[intervals < np.quantile(intervals, .80)]
        times_wo_outliers = times[intervals < np.quantile(intervals, .80)]
        # Plot final histogram of intervals
        plt.clf()
        interval_hist(ints_wo_outliers, PREF)

        # Plot scatter of intervals vs time
        plt.clf()
        time_scatter_plot(times_wo_outliers, ints_wo_outliers, PREF)

        # Get paths of images with charts
        hist_and_box = glob.glob(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh_w_hist/*_annot_w_hist.jpg")
        hist_and_box.sort() # Sort to correct order
        all_ims = [] # List to store image arrays
        for i in hist_and_box:
            all_ims.append(plt.imread(i).copy()) # Add arrays to list
        # Save video with charts and annotated images
        if LOG_FILE != None:
            int_exp_time = time.clock()
        imageio.mimsave(F"./model_annots/{PREF}annot_{CONF_PER}pc_thresh_w_hist.mp4", all_ims, fps=15)
        if LOG_FILE != None:
            vid_exp_time = time.clock()
    if LOG_FILE != None:
        with open(LOG_FILE, "a+") as lfile:
            lfile.write(F"Load detection graph : {d_graph-start}\nLabel and category   : {l_and_c-d_graph}\nBox calculation      : {box_time-l_and_c}\n")
            lfile.write(F"Create interval data : {int_create_time-box_time}\nExport interval data : {int_exp_time-int_create_time}\n")
            if not CSV_ONLY:
                lfile.write(F"Video export time: {vid_exp_time-int_exp_time}")

def use_model_multiple(PREFS, PATH_TO_CKPT='./training/frozen_inference_graph_v4.pb',
    PATH_TO_LABELS='./annotations/label_map.pbtxt', PATH_TO_IMS = './test_ims/',
    PATH_TO_ANNOT_IMS='./model_annots/', CSV_ONLY=False, FRAME_HEIGHT=989.9,
    FRAME_WIDTH=1319.9, FRAME_TIME=1.0, CONF_THR=0.3, OUTLIER_PROP=0.80,
    NUM_CLASSES=1, PATHS_TO_CSVS=None, SPEED_DAT_CSVS=None, LOG_FILE=None, SPLIT=None,
    FILE_EXT="jpg"):

    '''
    Args:
        PREFS: Image file prefix list.
        PATH_TO_CKPT: Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_LABELS: List of the strings that is used to add correct label for each box.
        PATH_TO_IMS: Path to image files.
        PATH_TO_ANNOT_IMS: Path to directory to store annotated images.
        CSV_ONLY: True if only comma-seperated value file outputs is desired.
        FRAME_HEIGHT: Frame HEIGHT in um, depends on microscope and magnification.
        FRAME_WIDTH: Frame width in um, depends on microscope and magnification.
        FRAME_TIME: Minutes between frames.
        CONF_THR: Confidence threshold to use for annotations, as a float.
        OUTLIER_PROP: Proportion of distances above which are considered outliers.
        NUM_CLASSES: Number of classes to detect
        PATHS_TO_CSVS: Path to exported CSVs of box annotations, as a list. Must correspond to order of PREFS.
        SPEED_DAT_CSVS: Name for speed data files, as a list. Must correspond to order of PREFS.
        LOG_FILE: Name for log file for timing data.
        SPLIT: Int (None) - If not none, segments image into SPLIT^2 images before prediction.
        FILE_EXT: str ('jpg') File extension of images used for annotation. Must be readable by matplotlib.plyplot.imread().
    '''

    if LOG_FILE != None:
        d_graph, l_and_c, box_time, int_create_time, int_exp_time, vid_exp_time = 0., 0., 0., 0., 0., 0.
        start = time.clock()
    CONF_PER = int(100 * CONF_THR)
        # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    if LOG_FILE != None:
        d_graph = time.clock()
    # Loading label map
    # Label maps map indices to category names
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    if LOG_FILE != None:
        l_and_c = time.clock()
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            if PATHS_TO_CSVS == None:
                PATHS_TO_CSVS = []
                for PREF in PREFS:
                    PATH_TO_CSVS.append(F"box_data_{PREF}.csv")
            for PATH_TO_CSV, PREF in zip(PATHS_TO_CSVS, PREFS):
                if not os.path.exists(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh/") and not CSV_ONLY:
                    os.mkdir(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh/")
                elif not CSV_ONLY:
                        print("Overwritting annotated images")
                if not os.path.exists(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh_w_hist/") and not CSV_ONLY:
                    os.mkdir(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh_w_hist/")
                with open(PATH_TO_CSV, "w") as file:
                    file.write("Frame,box1,box2,box3,box4,score,class\n")
                    test_ims = glob.glob(F"{PATH_TO_IMS}{PREF}*.{FILE_EXT}")
                    test_ims.sort()
                    all_ims = []
                    for i in test_ims:
                        print(i)
                        # Read frame from camera
                        image_np = plt.imread(i).copy()
                        if len(image_np.shape) == 2 or image_np.shape[2] == 1:
                            image_np = np.stack((image_np,)*3, axis=-1)
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        if SPLIT == None:
                            # Extract image tensor
                            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                            # Extract detection boxes
                            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                            # Extract detection scores
                            scores = detection_graph.get_tensor_by_name('detection_scores:0')
                            # Extract detection classes
                            classes = detection_graph.get_tensor_by_name('detection_classes:0')
                            # Extract number of detectionsd
                            num_detections = detection_graph.get_tensor_by_name(
                                'num_detections:0')
                            # Actual detection.
                            (boxes, scores, classes, num_detections) = sess.run(
                                [boxes, scores, classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})
                            # Visualization of the results of a detection.
                            buffer = ""
                            for x in range(np.squeeze(boxes).shape[0]):
                                buffer += F"{os.path.basename(i).split('.')[0]},{np.squeeze(boxes)[x,0]},{np.squeeze(boxes)[x,1]},{np.squeeze(boxes)[x,2]},{np.squeeze(boxes)[x,3]},{np.squeeze(scores)[x]},{np.squeeze(classes)[x]}\n"
                            file.write(buffer)
                            if not CSV_ONLY:
                                vis_util.visualize_boxes_and_labels_on_image_array(
                                    image_np,
                                    np.squeeze(boxes),
                                    np.squeeze(classes).astype(np.int32),
                                    np.squeeze(scores),
                                    category_index,
                                    use_normalized_coordinates=True,
                                    line_thickness=4,
                                    max_boxes_to_draw=100,
                                    min_score_thresh=CONF_THR)
                                save_image(image_np, F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh/{os.path.basename(i).split('.')[0]}_annot.jpg")
                                all_ims.append(image_np)
                        else:
                            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                            spl_fac = 1/SPLIT
                            im_segmented_list = segment_image(SPLIT, image_np_expanded)
                            boxes_all, scores_all, classes_all = [np.empty((1,0,4)), np.empty((1,0)), np.empty((1,0))]
                            for r, c, im_seg in zip(np.repeat(np.arange(SPLIT), SPLIT), np.tile(np.arange(SPLIT), SPLIT), im_segmented_list):
                                # Actual detection.
                                boxes_seg = detection_graph.get_tensor_by_name('detection_boxes:0')
                                # Extract detection scores
                                scores_seg = detection_graph.get_tensor_by_name('detection_scores:0')
                                # Extract detection classes
                                classes_seg = detection_graph.get_tensor_by_name('detection_classes:0')
                                # Extract number of detectionsd
                                num_detections_seg = detection_graph.get_tensor_by_name(
                                    'num_detections:0')
                                buffer = ""
                                (boxes_seg, scores_seg, classes_seg, num_detections_seg) = sess.run(
                                    [boxes_seg, scores_seg, classes_seg, num_detections_seg],
                                    feed_dict={image_tensor: im_seg})
                                # Visualization of the results of a detection.
                                buffer = ""
                                for x in range(np.squeeze(boxes_seg).shape[0]):
                                    buffer += F"{os.path.basename(i).split('.')[0]},{np.squeeze(boxes_seg)[x,0]*spl_fac + r*spl_fac},{np.squeeze(boxes_seg)[x,1]*spl_fac + c*spl_fac},"
                                    buffer += F"{np.squeeze(boxes_seg)[x,2]*spl_fac + r*spl_fac},{np.squeeze(boxes_seg)[x,3]*spl_fac + c*spl_fac},{np.squeeze(scores_seg)[x]},{np.squeeze(classes_seg)[x]}\n"
                                file.write(buffer)
                                boxes_seg = np.stack((boxes_seg[:,:,0]*spl_fac + r*spl_fac,
                                                            boxes_seg[:,:,1]*spl_fac + c*spl_fac,
                                                            boxes_seg[:,:,2]*spl_fac + r*spl_fac,
                                                            boxes_seg[:,:,3]*spl_fac + c*spl_fac), axis=-1)
                                boxes_all = np.hstack((boxes_all, boxes_seg))
                                scores_all = np.hstack((scores_all, scores_seg))
                                classes_all = np.hstack((classes_all, classes_seg))
                            if not CSV_ONLY:
                                print("Boxes shape, ", boxes_all.shape, classes_all.shape, scores_all.shape)
                                vis_util.visualize_boxes_and_labels_on_image_array(
                                    image_np,
                                    np.squeeze(boxes_all),
                                    np.squeeze(classes_all).astype(np.int32),
                                    np.squeeze(scores_all),
                                    category_index,
                                    use_normalized_coordinates=True,
                                    line_thickness=4,
                                    max_boxes_to_draw=100*SPLIT**2,
                                    min_score_thresh=CONF_THR)
                                save_image(image_np, F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh/{os.path.basename(i).split('.')[0]}_annot.jpg")
                                all_ims.append(image_np)
    if SPEED_DAT_CSVS == None:
        SPEED_DAT_CSVS = []
        for PREF in PREFS:
            SPEED_DAT_CSVS.append(F"{PREF}speed_data.csv")
    for PATH_TO_CSV, SPEED_DAT_CSV, PREF in zip(PATHS_TO_CSVS, SPEED_DAT_CSVS, PREFS):
        if LOG_FILE != None:
            box_time = time.clock()
        # Saves annotated frames to a .mp4 file
        if not CSV_ONLY:
            imageio.mimsave(F"./model_annots/{PREF}annot_{CONF_PER}pc_thresh.mp4", all_ims, fps=15)                # Display output

        ims = glob.glob(F"{PATH_TO_IMS}{PREF}*.{FILE_EXT}") # Gets list of all saved images
        ims.sort() # Sorts alphabetically
        ims_base = [] # List to store basenames without extension
        for i in ims:
            ims_base.append(os.path.basename(i).split(".")[0])

        intervals = np.array([]) # array to store distance intervals between tips
        times = np.array([]) # array to store times for each interval
        mid_y_arr, mid_x_arr = np.array([]), np.array([]) # x and y coords midpoint of tip-connecting segment
        dy_comps, dx_comps = np.array([]), np.array([]) # normalized x and y components of tip movement
        medians = [] # List to store median distance for each frame
        box_dat = pd.read_csv(PATH_TO_CSV) # Box position data, also tip position data

        # Extract center coordinates of boxes
        box_dat["ycoord"] = (box_dat.box1 + box_dat.box3)/2*(FRAME_HEIGHT)
        box_dat["xcoord"] = (box_dat.box2 + box_dat.box4)/2*(FRAME_WIDTH)

        # Select frame which meet these criterea: 1) the first frame and 2) are above confidence threshold
        box_dat_sub1 = np.array(box_dat[(box_dat.Frame == ims_base[0]) & (box_dat.score > CONF_THR)].iloc[:,7:])

        if CSV_ONLY:
            for i in range(len(ims_base)-1): # For each frame
                box_dat_sub2 = np.array(box_dat[(box_dat.Frame == ims_base[i+1]) & (box_dat.score > CONF_THR)].iloc[:,7:]) # Extract the next frame's box data
                if (box_dat_sub1.shape[0] > 0) & (box_dat_sub2.shape[0] > 0):
                    min_dists, norm_dy, norm_dx, mid_y, mid_x = calc_velocity(box_dat_sub1, box_dat_sub2)
                    intervals = np.concatenate((intervals, min_dists)) # Add minimum distances to intervals array
                    mid_y_arr = np.concatenate((mid_y_arr, mid_y))
                    mid_x_arr = np.concatenate((mid_x_arr, mid_x))
                    dy_comps = np.concatenate((dy_comps, norm_dy))
                    dx_comps = np.concatenate((dx_comps, norm_dx))
                    times = np.concatenate((times, np.repeat(i*FRAME_TIME, len(min_dists)))) # Add frame number to times
                box_dat_sub1 = box_dat_sub2
            if LOG_FILE != None:
                int_create_time = time.clock()
            # Create dataframe to store output
            speed_dat = pd.DataFrame({"Time" : times, "Speed" : intervals, "Y_component" : dy_comps, "X_component" : dx_comps, "Y_mid" : mid_y_arr, "X_mid" : mid_x_arr})
            # Export dataframe as CSV file
            speed_dat.to_csv(SPEED_DAT_CSV)
            if LOG_FILE != None:
                int_exp_time = time.clock()

        else:
            for i in range(len(ims_base)-1): # For each frame
                im2 = plt.imread(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh/{ims_base[i+1]}_annot.jpg") # Read in the next frame as an array
                box_dat_sub2 = np.array(box_dat[(box_dat.Frame == ims_base[i+1]) & (box_dat.score > CONF_THR)].iloc[:,7:]) # Extract the next frame's box data
                if (box_dat1.shape[0] > 0) & (box_dat2.shape[0] > 0):
                    min_dists, norm_dy, norm_dx, mid_y, mid_x = calc_velocity(box_dat_sub1, box_dat_sub2)

                    intervals = np.concatenate((intervals, min_dists)) # Add minimum distances to intervals array
                    mid_y_arr = np.concatenate((mid_y_arr, mid_y))
                    mid_x_arr = np.concatenate((mid_x_arr, mid_x))
                    dy_comps = np.concatenate((dy_comps, norm_dy))
                    dx_comps = np.concatenate((dx_comps, norm_dx))

                    times = np.concatenate((times, np.repeat(i*FRAME_TIME, len(min_dists)))) # Add frame number to times
                    if LOG_FILE != None:
                        int_create_time = time.clock()
                    ints_wo_outliers = intervals[intervals < np.quantile(intervals, OUTLIER_PROP)] # Remove top proportion as outliers
                medians.append(np.median(ints_wo_outliers)) # Store median of distances

                plt.clf() # Clear figure
                # Set up figure with subplots
                fig = plt.figure()
                gs = fig.add_gridspec(3,2)
                ax1 = fig.add_subplot(gs[0,0])
                ax2 = fig.add_subplot(gs[0,1])
                ax3 = fig.add_subplot(gs[1:3,:])
                # Plot median line chart
                ax1.plot(medians)
                ax1.set_ylabel("Median tip speed (um/min)")
                ax1.set_xlabel("Frame number")
                ax1.set_title("Median hyphal tip speed")
                # Plot histogram of tip speeds/intervals
                ax2.hist(ints_wo_outliers, bins = 30)
                ax2.set_xlabel("Tip speed (um/min)")
                ax2.set_ylabel("Count")
                ax2.set_title("Distribution of hyphal tip speeds")
                ax2.axis("tight")
                # Show annotated images
                ax3.imshow(im2)
                ax3.axis("off")
                plt.tight_layout()
                # Save the annotated figure
                plt.savefig(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh_w_hist/{ims_base[i+1]}_annot_w_hist.jpg", dpi=400)
                # Next frame
                box_dat_sub1 = box_dat_sub2
            # Create dataframe to store output
            speed_dat = pd.DataFrame({"Time" : times, "Speed" : intervals, "Y_component" : dy_comps, "X_component" : dx_comps, "Y_mid" : mid_y_arr, "X_mid" : mid_x_arr})
            # Export dataframe as CSV file
            speed_dat.to_csv(SPEED_DAT_CSV)
            # Slice the intervals and time array to remove "outliers"
            ints_wo_outliers = intervals[intervals < np.quantile(intervals, .80)]
            times_wo_outliers = times[intervals < np.quantile(intervals, .80)]
            # Plot final histogram of intervals
            plt.clf()
            interval_hist(ints_wo_outliers, PREF)

            # Plot scatter of intervals vs time
            plt.clf()
            time_scatter_plot(times_wo_outliers, ints_wo_outliers, PREF)

            # Get paths of images with charts
            hist_and_box = glob.glob(F"{PATH_TO_ANNOT_IMS}{PREF}annot_{CONF_PER}pc_thresh_w_hist/*_annot_w_hist.jpg")
            hist_and_box.sort() # Sort to correct order
            all_ims = [] # List to store image arrays
            for i in hist_and_box:
                all_ims.append(plt.imread(i).copy()) # Add arrays to list
            # Save video with charts and annotated images
            if LOG_FILE != None:
                int_exp_time = time.clock()
            imageio.mimsave(F"./model_annots/{PREF}annot_{CONF_PER}pc_thresh_w_hist.mp4", all_ims, fps=15)
            if LOG_FILE != None:
                vid_exp_time = time.clock()
        if LOG_FILE != None:
            with open(LOG_FILE, "a+") as lfile:
                lfile.write(F"Load detection graph : {d_graph-start}\nLabel and category   : {l_and_c-d_graph}\nBox calculation      : {box_time-l_and_c}\n")
                lfile.write(F"Create interval data : {int_create_time-box_time}\nExport interval data : {int_exp_time-int_create_time}\n")
                if not CSV_ONLY:
                    lfile.write(F"Video export time: {vid_exp_time-int_exp_time}")
