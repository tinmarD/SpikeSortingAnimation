import os, re
import numpy as np
import csv
import mne


def createuniquedir(dirpath):
    """ Create a unique directory from a path. If .../.../dir_name already exists, .../.../dir_name_2 is returned.

    Parameters
    ----------
    dirpath : str
        Directory path

    Returns
    -------
    unique_dir_path : str
        Unique directory path

    """
    if os.path.exists(dirpath):
        dirpath = dirpath + '_2'
    inc = 3
    while os.path.exists(dirpath):
        dirpath = re.sub('_\d+$', '_{}'.format(inc), dirpath)
        inc += 1
    os.mkdir(dirpath)
    return dirpath


def find_original_time(x_target, filename, keep_start, keep_end):
    """ From the Spyking-Circus spike time, get the original time in the original EEG file. This EEG file was first cut
    to remove artefacts. Start end end time of artefact-free periods are stored in arguments ``keep_start`` and
    ``keep_end``. Multiples files may have been used to do the spike-sorting, thus ``filename`` contains the filename
    for each artefact-free period.

    Parameters
    ----------
    x_target : float
        Time of the Spyking-Circus spike
    filename : array
        Filename for each artefact-free period
    keep_start : array
        Start time (s) of each artefact-free period
    keep_end : array
        End time (s) of each artefact-free period

    Returns
    -------
    y_ori : float
        Time of the spike in the original EEG file
    filename_i : str
        Name of the file containing the spike

    """
    if not np.isscalar(x_target):
        raise ValueError('Argument x_target must be a scalar')
    filename, keep_start, keep_end = np.array(filename), np.array(keep_start), np.array(keep_end)
    if not keep_start.size == keep_end.size == filename.size:
        raise ValueError('Columns filename and keep_start and keep_end must have the same size')
    n_rows = keep_start.size
    last_keep_end, i = 0, 0
    y_i, x_i, x = 0, 0, 0
    last_filename = filename[0]
    while 1:
        filename_i = filename[i]
        # New file :
        if not filename_i == last_filename:
            # print('New filename : {}'.format(filename[i]))
            last_keep_end = 0
            x = x + x_i
            x_i, y_i = 0, 0
        keep_start_i, keep_end_i = keep_start[i], keep_end[i]
        duration_i = keep_end_i - keep_start_i
        y_i += keep_start_i - last_keep_end
        if (x + x_i + duration_i) >= x_target:
            y_i += (x_target - (x + x_i))
            break
        else:
            x_i += duration_i
            y_i += duration_i
        last_keep_end = keep_end_i
        last_filename = filename_i
        i += 1
        if i >= n_rows:
            # raise ValueError('Could not reach target x : {}s'.format(x_target))
            print('Could not reach target x : {}s'.format(x_target))
            filename_i = []
            break
    return y_i, filename_i


def construct_artefact_free_edf_signal(sig_dirpath, interest_csv_filepath, sep=';', out_dirpath=[]):
    """ From a csv file containing the periods of interest (free of artefacts), construct an EDF file without the
    artefact periods.
    The output files are then synchronized with the SpykingCircus spike times.

    Parameters
    ----------
    sig_dirpath : str
        Path of the directory containing the EDF signals
    interest_csv_filepath : str
        Path of the file containing the periofs of interest (no artefact). This file must be a 3 column csv files.
        1st column contain the input EDF filename (in ``sig_dirpath``), the 2nd column contain the starting time (s)
        of the artefact, the 3rd colum contain the ending time of the artefact. Each row represents 1 artefact.
        1-line header.
    sep : str (default: ';')
        Csv file delimiter
    out_dirpath : str | none (default)
        Path of the directory where the output files are saved

    Returns
    -------

    """
    if not out_dirpath:
        out_dirpath = sig_dirpath
    # Open the file containing the period without artefacts
    with open(interest_csv_filepath, 'r') as f:
        reader = csv.reader(f, delimiter=sep)
        # Header must be 1 line
        reader.__next__()
        # Read all lines
        rows = []
        for row in reader:
            if not len(row) == 3:
                print('Row should have 3 columns')
            else:
                rows.append(row)
        rows = np.array(rows)
        n_rows = rows.shape[0]
        previous_edf_filename = ''
        raw_data = []
        t_start, i_part = 0, 0
        for i in range(n_rows):
            edf_filename_i = rows[i, 0]
            if not edf_filename_i == previous_edf_filename:  # New sig
                raw_sig_i = mne.io.read_raw_edf(os.path.join(sig_dirpath, edf_filename_i))
                fs = raw_sig_i.info['sfreq']
            # t_noart_start_i and t_noart_end_i are the starting/ending times of the artefact-free period
            t_noart_start_i, t_noart_end_i = float(rows[i, 1].replace(',', '.')), float(rows[i, 2].replace(',', '.'))
            # Read the data of the artefact free period
            noart_data_i = raw_sig_i.get_data(start=int(np.round(fs*t_noart_start_i)),
                                              stop=int(np.round(fs*t_noart_end_i)))
            raw_data.append(noart_data_i)
            # Write the file if current line is the last one or if a new signal is one the next line
            # if i == n_rows-1 or not rows[i+1, 0] == edf_filename_i:
            #     raw_data = np.hstack(raw_data)
            #     # n_pnts = raw_data.shape[1]
            #     # times = np.linspace(0, n_pnts / fs, n_pnts)
            #     raw_merged = mne.io.RawArray(data=raw_data, info=raw_sig_i.info, first_samp=0)
            #     # Write edf file
            #     t_end = t_start + raw_merged.n_times / fs
            #     # raw_merged.save(os.path.join(out_dirpath, 'part{}_{}_{}s.fif'.format(i_part, t_start, t_end)))
            #     t_start += raw_merged.n_times / fs
            #     # i_part += 1
            #     # raw_data = []




if __name__ == "__main__":
    interest_csv_filepath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\Artefact_EDF_reconstruction.csv'
    sig_dirpath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\signal_EDF_30kHz\Day1_20.03.15\monopolaire_30kHz'
    out_dirpath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\signal_EDF_30kHz\Day1_20.03.15\Sync'
    fs = 30000

    construct_artefact_free_edf_signal(sig_dirpath, interest_csv_filepath, out_dirpath=out_dirpath)
