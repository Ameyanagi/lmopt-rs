#!/usr/bin/env python3

with open('src/models/peak.rs', 'r') as f:
    content = f.read()

# Replace prefix in move |params, x| blocks with eval_prefix
in_eval_block = False
new_content = []
for line in content.split('\n'):
    if 'move |params, x|' in line:
        in_eval_block = True
    
    if in_eval_block and 'format!(' in line and ', prefix)' in line:
        line = line.replace(', prefix)', ', eval_prefix)')
    
    if in_eval_block and line.strip() == '});' or line.strip() == '})':
        in_eval_block = False
    
    new_content.append(line)

# Replace prefix in move |params, x, y| blocks with guess_prefix
content = '\n'.join(new_content)
in_guess_block = False
new_content = []
for line in content.split('\n'):
    if 'move |params, x, y|' in line:
        in_guess_block = True
    
    if in_guess_block and 'format!(' in line and ', prefix)' in line:
        line = line.replace(', prefix)', ', guess_prefix)')
    
    if in_guess_block and line.strip() == '});' or line.strip() == '})':
        in_guess_block = False
    
    new_content.append(line)

# Replace prefix in move |params, x| jacobian blocks with jac_prefix
content = '\n'.join(new_content)
in_jac_block = False
new_content = []
for line in content.split('\n'):
    if '.with_jacobian(' in line:
        in_jac_block = True
    
    if in_jac_block and 'format!(' in line and ', prefix)' in line:
        line = line.replace(', prefix)', ', jac_prefix)')
    
    if in_jac_block and line.strip() == ');':
        in_jac_block = False
    
    new_content.append(line)

with open('src/models/peak.rs', 'w') as f:
    f.write('\n'.join(new_content))