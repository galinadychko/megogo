import os
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
from multiprocessing import Pool
import  multiprocessing


class MML:

    def __init__(self, cache_folder):
        """
        Init class object parameter
        :param cache_folder: string: the full path to the folder for cache producing
        :param folder_to_install: string: the full path to the folder for MyMediaLite instalation
        """
        self.cache_folder=cache_folder

    def download(self, folder_to_install):
        """
        Download zip file from the official MyMediaLite website,
        create "MyMediaLite" folder
        :return: None
        """
        #os.system("if [ -d " + self.cache_folder+"/"+ folder_to_install + " ]; then rm -rf "
        #          + self.cache_folder+"/"+folder_to_install+" && mkdir -p "
        #          + self.cache_folder+"/"+folder_to_install+"; fi")
        os.system("if [ ! -d "+self.cache_folder+"/"+ folder_to_install
                  +" ]; then mkdir -p " + self.cache_folder+"/"+folder_to_install+"; fi")
        os.system("cd " + self.cache_folder + "/" + folder_to_install
                  +" && mkdir MyMediaLite")
        os.system("cd " + self.cache_folder + "/" + folder_to_install
                  + " && wget http://mymedialite.net/download/MyMediaLite-3.11.src.tar.gz")
        os.system("cd " + self.cache_folder + "/" + folder_to_install+
                  " && tar xf MyMediaLite-3.11.src.tar.gz")

    def replace(self, file_path, pattern, subst):
        """
        Replace the given pattern in the appropriate file by rhe given substitution
        :param file_path: string: the full path with name to the file
        :param pattern: string: the string, which should be changed
        :param subst: string: the substitution
        :return: None
        """
        fh, abs_path = mkstemp()
        with fdopen(fh,'w') as new_file:
            with open(file_path) as old_file:
                for line in old_file:
                    new_file.write(line.replace(pattern, subst))
        remove(file_path)
        move(abs_path, file_path)

    def install(self, folder_to_install):
        """
        Makke src files, replace appropriate string, make install
        :return: None
        """
        os.system("cd " + self.cache_folder + folder_to_install+"/"+"zenogantner-MyMediaLite-b9d0478"
                  " && make all")
        # os.system("cd " + self.cache_folder + "/" + self.folder_to_install+"/"+"zenogantner-MyMediaLite-b9d0478")
        self.replace(file_path=self.cache_folder + folder_to_install+"/"+"zenogantner-MyMediaLite-b9d0478"+"/Makefile",
                pattern="PREFIX=/usr/local",
                subst="PREFIX="+self.cache_folder+folder_to_install+"/MyMediaLite")
        os.system("cd " + self.cache_folder+folder_to_install+"/"+"zenogantner-MyMediaLite-b9d0478"
                  " && make install")

    def delete_temp(self, folder_to_install):
        """
        Delete temporary files: zip download and unzip folder
        :return: None
        """
        os.system("cd " + self.cache_folder + folder_to_install+
                  " && rm -rf zenogantner-MyMediaLite-b9d0478")
        os.system("cd " + self.cache_folder + folder_to_install+
                  " && rm -rf MyMediaLite-3.11.src.tar.gz")

    def make_install(self, folder_to_install):
        """
        Complete download MyMediaLite, install it and delete temporary files
        :return: None
        """
        self.download(folder_to_install)
        self.install(folder_to_install)
        self.delete_temp(folder_to_install)
        print("MyMediaLite was successfully installed")

    def make_install_in_parallel(self, max_threads):
        l = [str(elem) for elem in range(max_threads)]
        with Pool(max_threads) as p:
            p.map(self.make_install, l)
