import streamlit as st
import pandas as pd
from ortools.linear_solver import pywraplp
import plotly.express as px
from fpdf import FPDF
import graphviz

# --- PDF GENERATOR FUNCTION ---
def create_pdf_report(results_data, mode, cycle_time, headcount):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Line Balancing Action Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Optimization Goal: {mode}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Total Operators: {headcount} | Target/Optimal Cycle Time: {cycle_time:.2f} mins", ln=True, align='L')
    pdf.line(10, 35, 200, 35) 
    pdf.ln(10) 
    for station in results_data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 8, txt=f"{station['Workstation']} - {station['Operator Profile']}", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(200, 6, txt=f"Machines: {station['Machines Required']}", ln=True)
        pdf.cell(200, 6, txt=f"Operations: {station['Operations']}", ln=True)
        pdf.cell(200, 6, txt=f"Loaded Time: {station['Loaded Time (mins)']} mins", ln=True)
        pdf.ln(4) 
    return pdf.output(dest='S').encode('latin-1')

# --- OR-TOOLS SOLVER ENGINE ---
def run_solver(operations, sam, machines, machine_types, precedence, efficiency, num_workers, mode, target_cycle=None, max_mach=1):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver: return None, None, None, None, None

    operators = list(range(1, num_workers + 1))
    x, y = {}, {}
    
    for i in operations:
        for j in operators: x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')
    for m in machine_types:
        for j in operators: y[m, j] = solver.IntVar(0, 1, f'y_{m}_{j}')

    for i in operations: solver.Add(solver.Sum([x[i, j] for j in operators]) == 1)
    
    for before, after in precedence:
        solver.Add(solver.Sum([j * x[after, j] for j in operators]) >= solver.Sum([j * x[before, j] for j in operators]))
        
    for i in operations:
        for j in operators: solver.Add(x[i, j] <= y[machines[i], j])
            
    for j in operators:
        solver.Add(solver.Sum([y[m, j] for m in machine_types]) <= max_mach)

    if mode == "Type 1: Minimize Headcount":
        for j in operators:
            station_time = [(sam[i] / efficiency[i][j]) * x[i, j] for i in operations]
            solver.Add(solver.Sum(station_time) <= target_cycle)
        max_station = solver.IntVar(0, len(operators), 'max_station')
        for i in operations:
            for j in operators: solver.Add(max_station >= j * x[i, j])
        solver.Minimize(max_station)
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL: return status, x, y, int(max_station.SolutionValue()), target_cycle
            
    else: 
        c_max = solver.NumVar(0.0, solver.infinity(), 'c_max')
        for j in operators:
            station_time = [(sam[i] / efficiency[i][j]) * x[i, j] for i in operations]
            solver.Add(solver.Sum(station_time) <= c_max)
        solver.Minimize(c_max)
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL: return status, x, y, len(operators), c_max.SolutionValue()
            
    return status, None, None, None, None

# --- UI SETUP ---
st.set_page_config(page_title="Universal Line Balancer", layout="wide")
st.title("🏭 Universal Line Balancing Engine")
st.markdown("Upload data, review auto-generated precedence, and optimize.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Optimization Mode")
mode = st.sidebar.radio("What is your goal today?", ("Type 1: Minimize Headcount", "Type 2: Minimize Cycle Time", "Type E: Maximize Efficiency"))
st.sidebar.divider()
st.sidebar.header("Factory Constraints")
max_machines = st.sidebar.number_input("Max Machines per Station", min_value=1, value=1, step=1)

if mode == "Type 1: Minimize Headcount":
    target_cycle_time = st.sidebar.number_input("Target Cycle Time (mins)", min_value=0.1, max_value=200.0, value=1.5, step=0.1)
    max_allowed_headcount = st.sidebar.slider("Max Allowed Workers", 1, 100, 30)
elif mode == "Type 2: Minimize Cycle Time":
    fixed_workstations = st.sidebar.number_input("Fixed Number of Workers", min_value=1, max_value=100, value=25, step=1)
    shift_minutes = st.sidebar.number_input("Working Minutes per Shift", min_value=60, value=480, step=30)
else:
    min_w = st.sidebar.number_input("Minimum Workers", min_value=1, max_value=100, value=15, step=1)
    max_w = st.sidebar.number_input("Maximum Workers", min_value=2, max_value=100, value=35, step=1)

# --- STEP 1: UPLOAD & BULLETPROOF CLEANING ---
st.header("Step 1: Upload Excel/CSV")
uploaded_file = st.file_uploader("Upload your Operation Bulletin", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
        else: df = pd.read_excel(uploaded_file)
        
        # Ensure columns exist
        if 'Operation Description' not in df.columns and 'Operation' in df.columns:
            df = df.rename(columns={'Operation': 'Operation Description'})
            
        # 1. Force strings and remove completely empty/NaN rows
        df['Operation Description'] = df['Operation Description'].astype(str).str.strip().str.upper()
        df = df[~df['Operation Description'].isin(['', 'NAN', 'NONE', 'NULL'])]
        
        # 2. Force SAM to numeric, treating spaces/blanks as 0.0
        if 'SAM' in df.columns:
            df['SAM'] = pd.to_numeric(df['SAM'], errors='coerce').fillna(0.0)
        else:
            df['SAM'] = 0.0
            
        # --- PHASE 1: COLLECT BASE SUB-ASSEMBLY IDS ---
        base_sub_ids = []
        for index, row in df.iterrows():
            sam_val = row['SAM']
            desc = row['Operation Description']
            if sam_val == 0.0 and desc not in base_sub_ids:
                base_sub_ids.append(desc)
                        
        # --- PHASE 2: SMART AUTO-TAGGING ---
        current_sub_id = "General Assembly"
        valid_rows = []
        ignore_keywords = ["ASSEMBLY", "TAPE", "BINDING", "THE", "AND", "WITH"]
        
        for index, row in df.iterrows():
            sam_val = row['SAM']
            desc = row['Operation Description']
            
            # If SAM is 0.0, it's a Header
            if sam_val == 0.0:
                current_sub_id = desc
            else:
                assigned_id = current_sub_id
                
                # --- SMART MERGE LOGIC ---
                if "ASSEMBLE" in desc or "JOIN" in desc or "ATTACH" in desc:
                    for base_id in base_sub_ids:
                        if base_id == current_sub_id: continue 
                        
                        keywords = [k for k in base_id.replace('+', ' ').replace('/', ' ').split() if len(k) > 3 and k not in ignore_keywords]
                        match_found = False
                        for kw in keywords:
                            if kw in desc: match_found = True; break
                                
                        if match_found:
                            assigned_id = f"Merge: {current_sub_id} + {base_id}"
                            break 
                
                row['Sub-Assembly ID'] = assigned_id
                valid_rows.append(row)
                
        clean_df = pd.DataFrame(valid_rows)
        
        # --- STEP 2: REVIEW SUB-ASSEMBLY IDS ---
        st.divider()
        st.header("Step 2: Review Sub-Assembly Tags")
        st.markdown("The Smart Tagger assigned components and predicted Merges. **Click any cell in the 'Sub-Assembly ID' column to edit.**")
        
        edited_df = st.data_editor(clean_df, num_rows="dynamic", use_container_width=True)
        
        # --- AUTO-PRECEDENCE & MERGE LINKING LOGIC ---
        precedence_list = []
        
        for sub_id, group in edited_df.groupby('Sub-Assembly ID', sort=False):
            ops = group['Operation Description'].tolist()
            for i in range(len(ops)-1):
                precedence_list.append({"Before Operation": ops[i], "After Operation": ops[i+1]})
                
            if str(sub_id).startswith("Merge:"):
                parts = str(sub_id).replace("Merge:", "").split("+")
                for part in parts:
                    target_sub_id = part.strip()
                    target_group = edited_df[edited_df['Sub-Assembly ID'] == target_sub_id]
                    if not target_group.empty:
                        last_op = target_group['Operation Description'].iloc[-1]
                        first_merge_op = ops[0]
                        precedence_list.append({"Before Operation": last_op, "After Operation": first_merge_op})
        
        prec_df = pd.DataFrame(precedence_list).drop_duplicates()
        
        # --- STEP 3: REVIEW PRECEDENCE & FLOWCHART ---
        st.divider()
        st.header("Step 3: Review Precedence Flow")
        st.markdown("The system auto-linked sequences and merges. **Add or delete rows in the table below to fix any logic errors.**")
        
        # NEW OVERRIDE LOGIC
        st.info("💡 **Have a saved precedence table?** Upload it here to immediately override the auto-generated logic.")
        prec_upload = st.file_uploader("Upload Saved Precedence Table (CSV)", type=["csv"], key="prec_uploader")
        
        if prec_upload is not None:
            uploaded_prec = pd.read_csv(prec_upload)
            if 'Before Operation' in uploaded_prec.columns and 'After Operation' in uploaded_prec.columns:
                prec_df = uploaded_prec
                st.success("✅ Custom Precedence applied!")
            else:
                st.error("❌ The uploaded CSV must contain 'Before Operation' and 'After Operation' columns.")

        col_table, col_chart = st.columns([1, 1.5])
        
        with col_table:
            edited_prec_df = st.data_editor(prec_df, num_rows="dynamic", use_container_width=True)
            
            # Export button for saving custom configurations
            csv_prec = edited_prec_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download This Precedence Table", data=csv_prec, file_name="custom_precedence.csv", mime="text/csv", use_container_width=True)
        
        with col_chart:
            st.markdown("### Precedence Flowchart")
            dot = graphviz.Digraph()
            dot.attr(rankdir='TB', size='8,5')
            dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
            
            final_precedence_tuples = []
            for idx, row in edited_prec_df.iterrows():
                if pd.notna(row.get('Before Operation')) and pd.notna(row.get('After Operation')):
                    before_str = str(row['Before Operation']).strip()
                    after_str = str(row['After Operation']).strip()
                    if before_str and after_str:
                        dot.edge(before_str, after_str)
                        final_precedence_tuples.append((before_str, after_str))
                        
            st.graphviz_chart(dot)

        # --- STEP 4: OPTIMIZATION ENGINE ---
        st.divider()
        st.header("Step 4: Run Math Engine")
        
        if st.button("🚀 Confirm Flow & Run Optimization", type="primary"):
            
            # Final Safety Catch for User Inputs
            if mode == "Type 1: Minimize Headcount" and target_cycle_time is None: target_cycle_time = 1.5
            
            edited_df['SAM'] = pd.to_numeric(edited_df['SAM'], errors='coerce').fillna(0.0)
            operations = edited_df['Operation Description'].astype(str).tolist()
            sam = dict(zip(edited_df['Operation Description'], edited_df['SAM']))
            machines = dict(zip(edited_df['Operation Description'], edited_df['Machine'].fillna("MANUAL")))
            machine_types = list(set(machines.values()))
            total_sam = sum(sam.values())
            
            real_operator_cols = [col for col in edited_df.columns if col.startswith('Op_')]
            num_real_operators = len(real_operator_cols)
            
            efficiency = {op: {} for op in operations}
            for index, row in edited_df.iterrows():
                op_name = row['Operation Description']
                for j in range(1, 101):
                    if j <= num_real_operators:
                        col_name = real_operator_cols[j-1]
                        eff_value = float(row[col_name]) if pd.notna(row[col_name]) else 0.01 
                        efficiency[op_name][j] = eff_value
                    else:
                        efficiency[op_name][j] = 1.0

            final_status, final_x, final_y, final_headcount, final_cycle_time = None, None, None, None, None
            
            with st.spinner(f'Executing {mode} Mathematics...'):
                if mode == "Type 1: Minimize Headcount":
                    final_status, final_x, final_y, final_headcount, final_cycle_time = run_solver(
                        operations, sam, machines, machine_types, final_precedence_tuples, efficiency, max_allowed_headcount, mode, target_cycle_time, max_machines)
                elif mode == "Type 2: Minimize Cycle Time":
                    final_status, final_x, final_y, final_headcount, final_cycle_time = run_solver(
                        operations, sam, machines, machine_types, final_precedence_tuples, efficiency, fixed_workstations, mode, None, max_machines)

            if final_status == pywraplp.Solver.OPTIMAL:
                st.success(f"✅ Optimal Balance Found! Total Operators: **{final_headcount}** | Cycle Time: **{final_cycle_time:.2f} mins**")
                
                results_data = []
                chart_data = []
                cols = st.columns(3) 
                
                for j in range(1, final_headcount + 1):
                    assigned_ops = [i for i in operations if final_x[i, j].SolutionValue() > 0.5]
                    if not assigned_ops: total_time = 0.0; station_machines = []
                    else:
                        total_time = sum([sam[i] / efficiency[i][j] for i in assigned_ops])
                        station_machines = [m for m in machine_types if final_y[m, j].SolutionValue() > 0.5]
                        for op in assigned_ops: chart_data.append({"Workstation": f"Station {j}", "Operation": op, "Time (mins)": sam[op] / efficiency[op][j]})
                    
                    operator_type = f"Real Worker ({j})" if j <= num_real_operators else f"Standard Worker ({j})"
                    results_data.append({
                        "Workstation": f"Station {j}", "Operator Profile": operator_type,
                        "Machines Required": ", ".join(station_machines) if station_machines else "None",
                        "Operations": ", ".join(assigned_ops) if assigned_ops else "IDLE / EMPTY",
                        "Loaded Time (mins)": round(total_time, 2), "Cycle Limit": round(final_cycle_time, 2)
                    })

                    with cols[(j-1) % 3]: 
                        with st.container():
                            st.markdown(f"#### Station {j}")
                            if not assigned_ops: st.warning("⚠️ Station Idle")
                            else:
                                st.caption(f"👤 {operator_type}")
                                st.markdown(f"**Machines:** {', '.join(station_machines)}")
                                st.progress(min(total_time / final_cycle_time, 1.0))
                                st.markdown(f"**Load:** {total_time:.2f} / {final_cycle_time:.2f} mins")
                            st.divider()
                
                st.markdown("### 📊 Yamazumi Chart (Workload Distribution)")
                if chart_data:
                    df_chart = pd.DataFrame(chart_data)
                    fig = px.bar(df_chart, x="Workstation", y="Time (mins)", color="Operation", text="Operation")
                    fig.add_hline(y=final_cycle_time, line_dash="dash", line_color="red", annotation_text=f"Cycle Time: {final_cycle_time:.2f}m")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ The solver could not find a valid solution. Try checking precedence rules for deadlocks.")
    except Exception as e:
        st.error(f"An error occurred: {e}. Please ensure your file matches the template.")
