#!/usr/bin/env perl

$filename=$ARGV[0];
@lines=();
while(<STDIN>)
{
    next if /mca_btl_tcp_component_create_listen.*bind\(\) failed/;
    if ($filename eq "screen-output")
    {
        next unless /Surface love numbers:/;
    }
    push @lines, $_;
}

while (@lines && $lines[-1] =~ /^\s*$/)
{
    pop @lines;
}
print @lines;
