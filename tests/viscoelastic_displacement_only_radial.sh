#!/usr/bin/env perl

$filename=$ARGV[0];
@lines=();
while(<STDIN>)
{
    if ($filename eq "screen-output")
    {
        s/(Solving Stokes system \(AMG\)\.\.\.) \d+\+0 iterations\./$1 XYZ iterations./;
    }
    push @lines, $_;
}

while (@lines && $lines[-1] =~ /^\s*$/)
{
    pop @lines;
}
print @lines;
