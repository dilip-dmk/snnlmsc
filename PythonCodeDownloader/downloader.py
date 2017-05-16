import argparse
import fnmatch
import logging
import os
import posixpath
import socket
import sys
import timeit

from github import Github

if sys.version_info < (3, 0):
    # python 2
    import urlparse
    from urllib import urlretrieve
else:
    # python 3
    import urllib.parse as urlparse
    from urllib.request import urlretrieve

def main(args, loglevel):
    start = timeit.default_timer()
    filename = "files/"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    socket.setdefaulttimeout(args.timeout)
    execfile("/home/madhukum/pyenv/creds.py", globals())
    g = Github(username, password)
    with open(args.repo_file, 'r') as f:
        file_counter = 0
        for line in f.readlines():
            logging.info('Fetching repository: %s' % line)
            try:
                repo_str = line.rstrip().split('github.com/')[-1]
                repo = g.get_repo(repo_str, True)
                tree = repo.get_git_tree('master', recursive=True)
                files_to_download = []
                for file in tree.tree:
                    if fnmatch.fnmatch(file.path, args.wildcard):
                        files_to_download.append('https://github.com/%s/raw/master/%s' % (repo_str, file.path))
                for file in files_to_download:
                    logging.info('Downloading %s' % file)
                    file_counter += 1
                    filename = posixpath.basename(urlparse.urlsplit(file).path)
                    output_path = os.path.join(args.output_dir, filename)
                    if os.path.exists(output_path):
                        opath, ext = output_path.split('.')
                        output_path = opath + str(file_counter) + "." + ext
                    try:
                        urlretrieve(file, output_path)
                    except Exception as e:
                        logging.exception('Error downloading %s' % file)
            except Exception as e:
                logging.exception('Error fetching repository %s' % line)
    args.yara_meta = os.path.join(args.output_dir, args.yara_meta)
    with open(args.yara_meta, 'w') as f:
        for i in os.listdir(args.output_dir):
            try:
                f.write("include \"" + i + "\"\n")
            except Exception as e:
                logging.exception('Couldn\'t write to %s' % args.yara_meta)

    print "Number of py-files downloaded  : %d" % len(files_to_download)
    stop = timeit.default_timer()
    print "Elapsed time                : %8.2f seconds" % (stop - start)


# python downloader.py -r debug.txt -o files/

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Github py-files downloader")
    parser.add_argument("-r",
                        "--repo_file",
                        default="repos.txt",
                        help="Path for the input file which contains a url of a Github repository for each separate line")

    parser.add_argument("-w",
                        "--wildcard",
                        default="*.py",
                        help="Unix shell-style wildcard to match files to download (for example: *.txt)")

    parser.add_argument("-o",
                        "--output_dir",
                        default="files/",
                        help="Directory to store all downloaded files")

    parser.add_argument("-y",
                        "--yara-meta",
                        default="rules.yara",
                        help="Yara meta rule filename to create")

    parser.add_argument("-t",
                        "--timeout",
                        default=5000,
                        help="Socket timeout (seconds)")

    parser.add_argument("-v",
                        "--verbose",
                        help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    main(args, loglevel)
