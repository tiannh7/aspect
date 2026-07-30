#!/bin/bash

# Open MPI can emit this sandbox-specific warning before ASPECT starts. It is
# unrelated to the numerical result and is not present on regular CI hosts.
sed '/btl_tcp_component\.c:.*bind() failed/d' | sed '${/^$/d;}'
