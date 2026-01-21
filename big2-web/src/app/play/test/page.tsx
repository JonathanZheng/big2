import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export default function TestModePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-900 to-purple-950 flex flex-col items-center justify-center p-8">
      <div className="max-w-md w-full">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Test Mode</h1>
          <p className="text-purple-200">Coming Soon</p>
        </div>

        <Card className="bg-white/10 backdrop-blur border-purple-400/30">
          <CardHeader>
            <CardTitle className="text-white">Analyze Physical Games</CardTitle>
            <CardDescription className="text-purple-200">
              Get AI assistance for real-life Big 2 games
            </CardDescription>
          </CardHeader>
          <CardContent className="text-purple-100 space-y-3">
            <p>✓ Enter your 13-card hand</p>
            <p>✓ Get AI-suggested best moves</p>
            <p>✓ Track opponent plays</p>
            <p>✓ Real-time game state analysis</p>
          </CardContent>
        </Card>

        <div className="mt-8 text-center">
          <Link href="/">
            <Button variant="outline" className="border-purple-400 text-purple-100 hover:bg-purple-800">
              ← Back to Home
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
