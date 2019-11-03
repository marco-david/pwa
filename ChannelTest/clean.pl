use strict;
use warnings;
use v5.30.0;
use Cwd qw( abs_path getcwd cwd);
use File::Basename qw( dirname );
# use as: dirname(abs_path());
use experimental qw( switch );
use Term::ANSIColor;

sub cleanupimages {
  my $directory = cwd;
  print($directory,"\n");
  opendir my $dir, $directory or die("Failed to open path");
  my @files = readdir $dir;
  for (@files){
    if ($_ =~ /.png$/){
      print("$_ \n");
      rename $_, "backup_images/$_";
    }
}
}
cleanupimages();

