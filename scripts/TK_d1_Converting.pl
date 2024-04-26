

# Adjusting IMS.csv (such as decimal and range)
# perl this.pl --file [IMS.csv] --label -1 --unit 0.01 --machine imscope --save save.tsv --range min,max
# [in] spot_num,m/z_1,m/z_2, ...  <=> Soralix
# [in] X,Y,ROI,m/z_1,m/z_2, ...  <=> iMScopeβ
# [out] label spot_num  data *interval of m/z is defined by '-unit' (cumulatively rounded)
# [PID:v4]


use strict;
use warnings;
use Getopt::Long qw(:config posix_default no_ignore_case gnu_compat auto_help);
use Pod::Usage;
use IO::Handle;
#use Math::Round;
use Storable qw(dclone);

STDOUT->autoflush(1);

my $start_time = (time())[0];
print $0, "\n";

my $file;
my $label = -1;
my $unit = 0.01; #
my $save = 'save.txt';
my $range; #if unused, automatically rounded
my $machine; # solarix, imscope

my %opt = ();
GetOptions(\%opt, 'range=s' => \$range, 'file=s' => \$file, 'label=i' => \$label, 'unit=f' => \$unit, 'save=s' => \$save, 'machine=s' => \$machine, 'help|h');
pod2usage(1) if ($opt{help} or $opt{h});

if (!defined $label or !defined $file) {
    die "*** Option -file and -label required!\n";
}
#if (!defined $range) {
#    $range = "";
#}
my $range2 = "NA";
if(defined $range) {
	$range2 = $range;
}
print "FILE:	$file\nLABEL:	$label\nUNIT:	$unit\nRANGE:	$range2\n\n";


my ($range_hash , $min, $max, $cols, $d, $unit2) = @{&range_generating($range, $unit)}; #->{$m/z} = intensity
my @mz_order = sort {$a <=> $b} keys %{$range_hash};

my $mz_values; #caption
open my $f, '<', $file or die $!;
open my $out, '>', $save or die $!;
my $line = 0;

while (<$f>) {
    chomp;
    next if($_=~/^\#/ or $_=~/^$/);
    my @content = split /\,/, $_;

    @content = map {$_ =~ s/\s+//g; $_} @content;
    
    #header処理
    if($line == 0) {
        if ($_ =~ /^m\/z/ and $machine eq 'solarix') { ###
            shift @content;
            foreach(my $n=0; $n<@content; $n++) {
                $mz_values->[$n] = $content[$n]; #->{$n} = m/z
            }
            my $mz_count = @mz_order;
            print $out '#ORIGINAL_COLS:'.$cols.';MIN:'.$min.';MAX:'.$max.';OUT_COLS:'.$mz_count,"\n";
            print $out join "\t", ('#Label', 'Spot', @mz_order,"\n");
            $line++;
            next;
        } elsif($_=~ /^X/ and $machine eq 'imscope') {
            shift @content; # X
            shift @content; # Y
            shift @content; # ROI
            foreach(my $n=0; $n<@content; $n++) {
                $mz_values->[$n] = $content[$n]; #->{$n} = m/z
            }
            my $mz_count = @mz_order;
            print $out '#ORIGINAL_COLS:'.$cols.';MIN:'.$min.';MAX:'.$max.';OUT_COLS:'.$mz_count,"\n";
            print $out join "\t", ('#Label', 'Spot', @mz_order,"\n");
            $line++;
            next;
        } else {
            print "*** ERROR *** Please check the file format\n";
            exit();
        }
    }

    
    # intensity
    if($line) {
        my $spot;
        my $range_hash2 = dclone($range_hash);
        if($machine eq 'solarix') {
            ($spot = $content[0]) =~ s/^Spot//;
            shift @content;
        } elsif($machine eq 'imscope') {
            $spot = join ":", @content[0,1]; # 1_XXX:YYY
            $spot = $line.'_'.$spot;
            @content = @content[3..$#content];
        }      
        
        foreach(my $nn=0; $nn<@content; $nn++) {
            my $mzv = $mz_values->[$nn];            
            #$mzv = int($mzv*(10**$d))/(10**$d); #切り捨て
            $mzv = sprintf("%.${d}f", $mzv); #四捨五入
            if (exists $range_hash2->{$mzv}) {            
                if($content[$nn] > 0) {
                    #my $count = $range_hash2->{$mzv}{'count'};
                    #my $intensity = $range_hash2->{$mzv}{'mean'};                    
                    #$intensity = ($intensity*$count+$content[$nn])/($count+1);
                    #$range_hash2->{$mzv}{'count'} = $count+1;
                    #$range_hash2->{$mzv}{'mean'} = $intensity;
                    #print "::$mzv-$nn-$intensity-$count\n";
                    $range_hash2->{$mzv}{'sum'} += $content[$nn]; 
                    #print "::$mzv-$nn-$content[$nn]\n";
                    
                }
            }
        }
        
        
        # save
        my @intensities;
        for my $key (@mz_order) {
            push @intensities, $range_hash2->{$key}{'sum'};
        }
        print $out join "\t", ($label, $spot, @intensities,"\n");
    } else {
        next;
    }
    $line++;
}
close $f;
close $out;


sub range_generating {
    my $range = shift;
    my $unit = shift;
    my $hash;
    my ($min, $max);
    my $total_col;
    
    my ($d, $unit2) = @{&calculating($unit)}; #decimal,integerized value
    
    if ($range){
        ($min, $max) = split /\,/, $range;
        $total_col = ($max-$min)*(10**$d)+1;
    } else { # not applied
        if ($machine eq 'solarix') {
            $min = `head -15 $file | grep ^m/z | cut -d',' -f2`;
            $min =~ s/^\s+//;
            chomp $min;
            my $oneliner = 'head -15 '.$file.' | grep ^m/z | perl -lane \'@a=split /\,/, $_; $b = @a; print $b\'';
            $total_col = `$oneliner`;
            $total_col =~ s/^\s+//;
            $total_col--; ### 最初のcsv format => 要確認
            chomp $total_col;
            $max = `head -15 $file | grep ^m/z | cut -d',' -f$total_col`;
            $max =~ s/^\s+//;
            chomp $max;
        } elsif ($machine eq 'imscope') {
            $min = `head -15 $file | grep ^X | cut -d',' -f4`;
            $min =~ s/^\s+//;
            chomp $min;
            my $oneliner = 'head -15 '.$file.' | grep ^X | perl -lane \'@a=split /\,/, $_; $b = @a; print $b\'';
            $total_col = `$oneliner`;
            $total_col =~ s/^\s+//;
            chomp $total_col;
            $max = `head -15 $file | grep ^m/z | cut -d',' -f$total_col`;
            $max =~ s/^\s+//;
            $total_col = $total_col - 4; ###
            chomp $max;
        }

    }

    #$min = int($min*(10**$d)); #discard the decimal part
    #$max = int($max*(10**$d)); #discard the decimal part
    $min = sprintf("%.${d}f", $min)*(10**$d); #rounding
    $max = sprintf("%.${d}f", $max)*(10**$d); #rounding
    foreach(my $n=$min; $n<=$max; $n+=$unit2) { #including max
        my $origin = $n/(10**$d);
        #$hash->{$origin}{'count'} = 0;
        #$hash->{$origin}{'mean'} = 0;
        $hash->{$origin}{'sum'} = 0;
    }
    $min = $min/(10**$d);
    $max = $max/(10**$d);
    return [($hash,$min,$max,$total_col,$d,$unit2)];
}

sub calculating {
    my $unit = shift;
    my $d = 0;
    while ($unit<1) {
        $unit = $unit*10;
        $d++;
    } 
    return [($d, $unit)];
}


my $end_time = (time())[0];
my $process_time = $end_time - $start_time;
printf("Process time [%02d:%02d:%02d]\n", int($process_time/3600), int(($process_time%3600)/60), $process_time%60);
exit;

__END__
 
=pod
 
=head1 SYNOPSIS
 
B<perl [options] this_script.pl>
 
Options: [-h|--help]
 
=head1 OPTIONS
 
=over 8
 
=item B<-h|--help>
 
print help.
 
=back
 
=head1 DESCRIPTION

=cut
