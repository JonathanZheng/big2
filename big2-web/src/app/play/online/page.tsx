import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export default function PlayOnlinePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-900 to-blue-950 flex flex-col items-center justify-center p-8">
      <div className="max-w-md w-full">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Online Multiplayer</h1>
          <p className="text-blue-200">Coming Soon</p>
        </div>

        <Card className="bg-white/10 backdrop-blur border-blue-400/30">
          <CardHeader>
            <CardTitle className="text-white">Features in Development</CardTitle>
            <CardDescription className="text-blue-200">
              We&apos;re working hard to bring you online play
            </CardDescription>
          </CardHeader>
          <CardContent className="text-blue-100 space-y-3">
            <p>✓ Create private rooms with room codes</p>
            <p>✓ Invite friends to play</p>
            <p>✓ Random matchmaking queue</p>
            <p>✓ Player statistics and ELO ratings</p>
          </CardContent>
        </Card>

        <div className="mt-8 text-center">
          <Link href="/">
            <Button variant="outline" className="border-blue-400 text-blue-100 hover:bg-blue-800">
              ← Back to Home
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
