#!/usr/bin/env perl

$filename=$ARGV[0];
@lines=();
while(<STDIN>)
{
    next if $_ =~ m/^-- /;
    next if $_ =~ m/^\|/;
    push @lines, $_;
}

while (@lines
       && ($lines[-1] =~ /^\s*$/
           || $lines[-1] =~ /^[-+]+\s*$/))
{
    pop @lines;
}
print @lines;
