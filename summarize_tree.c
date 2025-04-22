#include <stdio.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <string.h>

static int num_dirs, num_regular;

bool is_dir(const char *path)
{
  /*
   * Use the stat() function (try "man 2 stat") to determine if the file
   * referenced by path is a directory or not.  Call stat, and then use
   * S_ISDIR to see if the file is a directory. Make sure you check the
   * return value from stat() in case there is a problem, e.g., maybe the
   * the file doesn't actually exist.
   */

  struct stat buf;
  int status = stat(path, &buf); // https://stackoverflow.com/questions/4553012/checking-if-a-file-is-a-directory-or-just-a-file
  if (status == 0)
  {
    return S_ISDIR(buf.st_mode);
  }
  else
  {
    perror("error in stat()");
    return false;
  }
}

/*
 * I needed this because the mutual recursion means there's no way to
 * order them so that the definitions all precede the cause.
 */
void process_path(const char *);

void process_directory(const char *path)
{
  /*
   * Update the number of directories seen, use opendir() to open the
   * directory, and then use readdir() to loop through the entries
   * and process them. You have to be careful not to process the
   * "." and ".." directory entries, or you'll end up spinning in
   * (infinite) loops. Also make sure you closedir() when you're done.
   *
   * You'll also want to use chdir() to move into this new directory,
   * with a matching call to chdir() to move back out of it when you're
   * done.
   */
  // num_dirs++;
  // chdir(path);
  // DIR* dir = opendir(path); // open the directory
  // struct dirent* child;

  // while((child = readdir(dir)) != NULL){ // if the next item in the Dir is not null assign it to `name`
  //   if(strcmp(child->d_name,".") + (strcmp(child->d_name,"..") != 0)){
  //     process_path(child->d_name);
  //   }
  // }
  // closedir(path);
  // chdir("..");

  num_dirs++;
  int chdir_err = chdir(path);
  if (chdir_err != 0)
  {
    perror("Error: change directory failed\n");
    fprintf(stderr, "Path: %s\n", path);
    exit(1);
  }
  DIR *dir = opendir(".");
  struct dirent *ent;
  while ((ent = readdir(dir)) != NULL)
  {
    if (strcmp(ent->d_name, "..") == 0 || strcmp(ent->d_name, ".") == 0)
    {
      continue;
    }
    process_path(ent->d_name);
  }
  closedir(dir);
  int second_chdir_err = chdir("..");
  if (second_chdir_err != 0)
  {
    fprintf(stderr, "Path: %s\n", path);
    exit(1);
  }
}

void process_file(const char *path)
{
  /*
   * Update the number of regular files.
   * This is as simple as it seems. :-)
   */
  num_regular++;
}

void process_path(const char *path)
{
  if (is_dir(path))
  {
    process_directory(path);
  }
  else
  {
    process_file(path);
  }
}

int main(int argc, char *argv[])
{
  // Ensure an argument was provided.
  if (argc != 2)
  {
    printf("Usage: %s <path>\n", argv[0]);
    printf("       where <path> is the file or root of the tree you want to summarize.\n");
    return 1;
  }

  num_dirs = 0;
  num_regular = 0;

  process_path(argv[1]);

  printf("There were %d directories.\n", num_dirs);
  printf("There were %d regular files.\n", num_regular);

  return 0;
}
