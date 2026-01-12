import fs from "fs";
import { parse } from "csv-parse/sync";
import { network } from "hardhat";


type Commitment = {
  commitment: string;
  id: string;
};

function loadCommitments(csvPath: string): Commitment[] {
  const csv = fs.readFileSync(csvPath);
  const records = parse(csv, {
    columns: true,
    skip_empty_lines: true,
  });

  return records.map((r: any) => ({
    commitment: r.commitment_hex.trim(), // 0x...
    id: r.subject_id.trim(),
  }));
}

function buildUsers(base: Commitment[], N: number) {
  const users = [];
  for (let i = 0; i < N; i++) {
    const b = base[i % base.length];
    users.push({
      commitment: b.commitment,
      did: `did:rppg:${b.id}:${i}`,
    });
  }
  return users;
}

async function main() {
  console.log("Connecting to network (Hardhat v3)...");
  const conn = await network.connect();
  const ethers = conn.ethers;

  console.log("Deploying PhysioChain...");
  const physio = await ethers.deployContract("PhysioChain");
  await physio.waitForDeployment();
  console.log("PhysioChain deployed at:", await physio.getAddress());

  console.log("Loading commitments from CSV...");
  const csv = fs.readFileSync("ubfc_npz_commitments.csv");
  const records = parse(csv, { columns: true, skip_empty_lines: true });

  const base = records.map((r: any) => ({
    commitment: r.commitment_hex.trim(),
    id: r.subject_id.trim(),
  }));

  console.log(`Loaded ${base.length} base commitments`);

  const Ns = [100, 500, 1000, 2000, 5000, 10000];
  const results: any[] = [];

  for (const N of Ns) {
    console.log(`\n=== Running register experiment N=${N} ===`);

    let totalGas = 0n;
    const t0 = Date.now();

    for (let i = 0; i < N; i++) {
      const b = base[i % base.length];
      const did = `did:rppg:${b.id}:${i}`;

      const tx = await physio.register(b.commitment, did);
      const receipt = await tx.wait();
      totalGas += receipt!.gasUsed;

      if ((i + 1) % 50 === 0 || i === N - 1) {
        console.log(`  ${i + 1}/${N} registered`);
      }
    }

    const elapsedMs = Date.now() - t0;
    const avgGas = Number(totalGas) / N;

    console.log(`Finished N=${N}: avgGas=${avgGas.toFixed(2)}, elapsed=${elapsedMs} ms`);
    results.push({ N, avgGas, elapsedMs });
  }

  fs.writeFileSync("gas_results.json", JSON.stringify(results, null, 2));
  console.log("\nSaved gas_results.json");
}


main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
