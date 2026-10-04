import { RaceView } from "./RaceView";

export default async function RacePage({ params }: PageProps<"/race/[round]">) {
  const { round } = await params;
  return <RaceView round={Number(round)} />;
}
