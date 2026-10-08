# Session 1 Quiz — Answer Key

Total points: 0  ·  15 questions  ·  2 attempts  ·  30 min

**Q1** (mc, 0 pt, Reading §3 · bucket) In S3, what is a <strong>bucket</strong>?
   Correct: A named container for objects; its name is unique across all of AWS

**Q2** (mc, 0 pt, Reading §3 · key) The weather file for 2024 is stored as <code>csv/by_year/2024.csv</code> in the bucket <code>noaa-ghcn-pds</code>. In S3 terms, what is <code>csv/by_year/2024.csv</code>?
   Correct: The object's key: its full name inside the bucket, one string

**Q3** (mc, 0 pt, Reading §3 · prefix) When S3 shows what looks like a folder called <code>csv/</code>, what is it really showing you?
   Correct: A prefix: the shared beginning of many keys

**Q4** (mc, 0 pt, Reading §1, §3 · storage vs computation) Which statement about S3 is correct?
   Correct: S3 stores data; it cannot run your code. Some computer has to read the data to do anything with it

**Q5** (mc, 0 pt, Reading, 'what a RUNS ON comment does' · where code runs) You open a notebook in JupyterLab inside your SageMaker space and run a cell. Which computer runs that cell?
   Correct: The SageMaker instance; your laptop is only showing a browser tab

**Q6** (mc, 0 pt, Reading, 'what a RUNS ON comment does') A cell begins with <code># RUNS ON: SageMaker (remote)</code>, but you run it in a notebook you opened in Positron on your laptop. What happens?
   Correct: It runs on your laptop. The comment is a label for people; it does not move the code

**Q7** (mc, 0 pt, Reading §1 · total vs available RAM) A laptop reports 16 GB of <strong>total</strong> RAM and 4 GB <strong>available</strong>. Which number decides whether a new dataset can be loaded, and why are they different?
   Correct: 4 GB; the browser, chat apps, and other running programs already hold the rest

**Q8** (mc, 0 pt, Reading §1 · disk vs RAM) A file is 3 GB on disk and your laptop has 500 GB of free disk space and 4 GB of available RAM. Why might <code>pd.read_csv</code> on the whole file still fail?
   Correct: pandas works in RAM, and a loaded DataFrame is often larger than the file; 4 GB of RAM may not be enough

**Q9** (mc, 0 pt, Reading §1 · cores and vCPUs) A cloud machine is sold as <strong>2 vCPUs, 4 GB</strong>. What is a vCPU?
   Correct: One logical CPU; two vCPUs may be a single physical core presenting itself as two

**Q10** (mc, 0 pt, Reading §2 · units) A drive sold as "1 TB" shows up as about 931 GB in your file browser. Why?
   Correct: The seller counts in thousands (decimal units) and the operating system counts in 1,024s (binary units)

**Q11** (mc, 0 pt, Reading §4 · anonymous data, your own account) NOAA's weather bucket can be read anonymously, without signing in. So why does this module require you to have your own AWS account?
   Correct: The account lets you rent the remote computer (the SageMaker space); the data itself is free to read

**Q12** (mc, 0 pt, Reading §4 · GHCN conventions) In the GHCN-Daily data, a <code>TMAX</code> value is stored as <code>266</code> and its quality flag is empty. What is the temperature, and what would you do if the quality flag were <strong>not</strong> empty?
   Correct: 26.6 °C (values are in tenths); a flagged value failed NOAA's checks and is dropped, leaving that day missing

**Q13** (ma, 0 pt, Reading §6 · cost (select all that apply)) <em>Select all that apply.</em> In the required Session 1 workflow, which of these cost you money (from your credits)?
   Correct: A SageMaker space whose status is <em>Running</em>, for every hour it is running | The small disk attached to your space, per GB per month, even when the space is stopped

**Q14** (ma, 0 pt, Reading, workflow table (select all that apply)) <em>Select all that apply.</em> In the eight-step workflow, which steps run code on the <strong>SageMaker</strong> machine?
   Correct: Reading NOAA data from S3 | Filtering, transforming, and aggregating the data | Saving a small summary CSV

**Q15** (ma, 0 pt, Reading §6 · shutdown checklist (select all that apply)) <em>Select all that apply.</em> You are finished for the day. Which of these actually stops the hourly charge for your SageMaker space?
   Correct: Clicking <strong>Stop space</strong> on the space's page and waiting for the status to read <em>Stopped</em> | Idle Shutdown turning the space off after 60 minutes with no activity
