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
st.markdown("Upload data, apply your Precedence CSV, and run the math.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Optimization Mode")
mode = st.sidebar.radio("What is your goal today?", ("Type 1: Minimize Headcount", "Type 2: Minimize Cycle Time"))
st.sidebar.divider()
st.sidebar.header("Factory Constraints")
max_machines = st.sidebar.number_input("Max Machines per Station", min_value=1, value=1, step=1)

if mode == "Type 1: Minimize Headcount":
    target_cycle_time = st.sidebar.number_input("Target Cycle Time (mins)", min_value=0.1, max_value=200.0, value=1.5, step=0.1)
    max_allowed_headcount = st.sidebar.slider("Max Allowed Workers", 1, 100, 30)
elif mode == "Type 2: Minimize Cycle Time":
    fixed_workstations = st.sidebar.number_input("Fixed Number of Workers", min_value=1, max_value=100, value=25, step=1)

# --- STEP 1: UPLOAD & BULLETPROOF CLEANING ---
st.header("Step 1: Upload Main SAM List")
uploaded_file = st.file_uploader("Upload your Operation Bulletin", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file, sep=None, engine='python')
        else: df = pd.read_excel(uploaded_file)
        
        if 'Operation Description' not in df.columns and 'Operation' in df.columns:
            df = df.rename(columns={'Operation': 'Operation Description'})
            
        df['Operation Description'] = df['Operation Description'].astype(str).str.strip().str.upper()
        df = df[~df['Operation Description'].isin(['', 'NAN', 'NONE', 'NULL'])]
        
        if 'SAM' not in df.columns: df['SAM'] = 0.0
        df['SAM'] = pd.to_numeric(df['SAM'], errors='coerce').fillna(0.0)
        if 'Machine' not in df.columns: df['Machine'] = "MANUAL"
        
        st.markdown("**Editable Operations List:**")
        # Hide the ugly index numbers
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True)
        
        # --- STEP 2: REVIEW PRECEDENCE & FLOWCHART ---
        st.divider()
        st.header("Step 2: Precedence Flow")
        st.info("💡 **Upload your saved Precedence Table (CSV) here.**")
        prec_upload = st.file_uploader("Upload Saved Precedence Table (CSV)", type=["csv"], key="prec_uploader")
        
        if prec_upload is not None:
            prec_df = pd.read_csv(prec_upload, sep=None, engine='python')
            if 'Before Operation' in prec_df.columns and 'After Operation' in prec_df.columns:
                st.success("✅ Custom Precedence applied!")
            else:
                st.error("❌ Invalid CSV format.")
        else:
            ops = edited_df['Operation Description'].tolist()
            prec_list = [{"Before Operation": ops[i], "After Operation": ops[i+1]} for i in range(len(ops)-1)] if len(ops)>1 else []
            prec_df = pd.DataFrame(prec_list, columns=["Before Operation", "After Operation"])

        col_table, col_chart = st.columns([1, 1.5])
        
        with col_table:
            # Hide the ugly index numbers here too!
            edited_prec_df = st.data_editor(prec_df, num_rows="dynamic", use_container_width=True, hide_index=True)
            csv_prec = edited_prec_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download This Precedence Table", data=csv_prec, file_name="custom_precedence.csv", mime="text/csv", use_container_width=True)
        
        with col_chart:
            st.markdown("### Precedence Flowchart")
            dot = graphviz.Digraph()
            dot.attr(rankdir='TB', size='8,5')
            dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
            
            final_precedence_tuples = []
            for idx, row in edited_prec_df.iterrows():
                before_str = str(row.get('Before Operation', '')).strip()
                after_str = str(row.get('After Operation', '')).strip()
                if before_str and after_str and before_str != 'nan' and after_str != 'nan':
                    dot.edge(before_str, after_str)
                    final_precedence_tuples.append((before_str, after_str))
                    
            st.graphviz_chart(dot)

        # --- STEP 3: OPTIMIZATION ENGINE ---
        st.divider()
        st.header("Step 3: Run Math Engine")
        
        if st.button("🚀 Confirm Flow & Run Optimization", type="primary"):
            operations = edited_df['Operation Description'].astype(str).tolist()
            sam = dict(zip(edited_df['Operation Description'], edited_df['SAM']))
            machines = dict(zip(edited_df['Operation Description'], edited_df['Machine'].fillna("MANUAL")))
            machine_types = list(set(machines.values()))
            
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

            valid_precedence = []
            missing_ops = set()
            for b, a in final_precedence_tuples:
                if b in operations and a in operations:
                    valid_precedence.append((b, a))
                else:
                    if b not in operations: missing_ops.add(b)
                    if a not in operations: missing_ops.add(a)
                    
            if missing_ops:
                st.warning(f"⚠️ **Warning:** The following operations in your Precedence Table do not exactly match the operations in your Step 1 data (check for typos). Their precedence rules were ignored: \n\n {', '.join(missing_ops)}")

            final_status, final_x, final_y, final_headcount, final_cycle_time = None, None, None, None, None
            
            with st.spinner(f'Executing Mathematics...'):
                if mode == "Type 1: Minimize Headcount":
                    final_status, final_x, final_y, final_headcount, final_cycle_time = run_solver(
                        operations, sam, machines, machine_types, valid_precedence, efficiency, max_allowed_headcount, mode, target_cycle_time, max_machines)
                elif mode == "Type 2: Minimize Cycle Time":
                    final_status, final_x, final_y, final_headcount, final_cycle_time = run_solver(
                        operations, sam, machines, machine_types, valid_precedence, efficiency, fixed_workstations, mode, None, max_machines)

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
                st.error("❌ The solver could not find a valid solution. Try checking precedence rules for deadlocks or increasing the target cycle time.")
    except Exception as e:
        st.error(f"An error occurred: {e}. Please ensure your file matches the template.")
