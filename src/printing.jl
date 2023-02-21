
function prettyvalue(x::Value)
    op_str = x.op=="" ? " " : x.op
    str = string(round(x.data,digits = 2)," (gr: ",round(x.grad,digits=2),", op: ",op_str,")")
    return str
end
Base.show(io::IO, x::Value) = print(io,"$(prettyvalue(x))")
Base.show(io::IO,m::MIME"text/plain", x::Value) = print(io,"$(prettyvalue(x))")

"""
    printgraph(nodes,depth)

Print a simple polytree of the computation graph in the terminal.

Please note only the first topological parent of a shared child will have a connection if shared children are present.
    - please file a PR if you can come up with a good way to show this in the terminal
"""
function printgraph(nodes,depth)
    md = maximum(depth)
    lines = repeat([""],md)
    edges = repeat([""],md)
    cursor=0
    
    n = reverse(nodes)
    d = reverse(depth)
    
    i = 1
    cur_line_old = 0
    while i<= length(n)
        cur_line = d[i]

        # handle padding between terms
        if cur_line > cur_line_old
            pad = max(0,cursor - length(lines[cur_line]))
            lines[cur_line] = lines[cur_line]*repeat(" ",pad)
        elseif cur_line < cur_line_old
            pad = max(0,length(lines[cur_line_old])-length(lines[cur_line]))
            lines[cur_line] = lines[cur_line]*repeat(" ",pad)
            cursor = length(lines[cur_line])
        else
            cursor = length(lines[cur_line])
        end
        
        # get the padding for the minimal tree drawing
        pad = max(0,length(lines[cur_line]) - length(edges[cur_line]))

        # add the term
        str = prettyvalue(n[i])
        lines[cur_line] = lines[cur_line]*str*"   "
        
        # add the tree
        if cur_line==cur_line_old || cur_line < cur_line_old
            edges[cur_line] = edges[cur_line]*repeat("-",pad)*"|"
        else
            edges[cur_line] = edges[cur_line]*repeat(" ",pad)*"|"
        end

        i +=1
        cur_line_old = cur_line
    end
    out = [(e,l) for (l,e) in zip(lines,edges)]
    out = collect(Iterators.flatten(out))

    print("\nSimple Polytree Viewer: \n\n",join(out[2:end],"\n"))
    return nothing
end